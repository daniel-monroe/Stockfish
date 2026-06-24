/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of input feature PawnStruct of NNUE evaluation function

#include "pawn_struct.h"

#include "pawn_vocab.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

struct Key {
    std::uint64_t a, b;
    bool operator==(const Key& o) const { return a == o.a && b == o.b; }
};
struct KeyHash {
    std::size_t operator()(const Key& k) const {
        std::uint64_t h = k.a * 0x9E3779B97F4A7C15ull;
        h ^= k.b + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        return std::size_t(h);
    }
};

// Color-fold one half-board: lex-min of (w,b) and (flip b, flip w).
inline Key canon(std::uint64_t w, std::uint64_t b) {
    std::uint64_t fw = __builtin_bswap64(b), fb = __builtin_bswap64(w);
    Key k1{w, b}, k2{fw, fb};
    return (k2.a < k1.a || (k2.a == k1.a && k2.b < k1.b)) ? k2 : k1;
}

using Map = std::unordered_map<Key, int, KeyHash>;

Map make_map(const std::uint64_t (*vocab)[2], std::size_t n) {
    Map m;
    m.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i)
        m[{vocab[i][0], vocab[i][1]}] = int(i);
    return m;
}

const Map& build_map(const std::uint64_t (*vocab)[2], std::size_t n) {
    // Build each half-board's vocabulary map exactly once, the first time it is
    // needed. Function-local statics are initialized in a thread-safe manner
    // ([stmt.dcl]/4): concurrent search threads that reach this point block until
    // the single initializing thread finishes, so the map can never be observed
    // half-built. (The previous lazy `if (m.empty()) fill` had a data race that
    // corrupted the map under multi-threaded search.)
    if (vocab == PAWN_VOCAB_LEFT)
    {
        static const Map left = make_map(PAWN_VOCAB_LEFT, n);
        return left;
    }
    static const Map right = make_map(PAWN_VOCAB_RIGHT, n);
    return right;
}

}  // namespace

int PawnStruct::left_index(Bitboard wp, Bitboard bp) {
    const Map& m = build_map(PAWN_VOCAB_LEFT, NumLeft);
    auto       it = m.find(canon(wp & LeftMask, bp & LeftMask));
    return it == m.end() ? -1 : it->second;
}

int PawnStruct::right_index(Bitboard wp, Bitboard bp) {
    const Map& m = build_map(PAWN_VOCAB_RIGHT, NumRight);
    auto       it = m.find(canon(wp & RightMask, bp & RightMask));
    return it == m.end() ? -1 : it->second;
}

void PawnStruct::indices_for(const Position& pos, IndexList& out) {
    const Bitboard wp = pos.pieces(WHITE, PAWN);
    const Bitboard bp = pos.pieces(BLACK, PAWN);
    int            l  = left_index(wp, bp);
    int            r  = right_index(wp, bp);
    if (l >= 0)
        out.push_back(IndexType(l));
    if (r >= 0)
        out.push_back(IndexType(NumLeft + r));
}

void PawnStruct::append_active_indices(Color, const Position& pos, IndexList& active) {
    indices_for(pos, active);  // perspective-symmetric
}

bool PawnStruct::pawn_structure_changed(const DiffType& diff) {
    if (type_of(diff.pc) == PAWN)
        return true;  // a pawn moved (or promoted: pc==PAWN, to==SQ_NONE)
    if (diff.remove_sq != SQ_NONE && type_of(diff.remove_pc) == PAWN)
        return true;  // a pawn was captured
    if (diff.add_sq != SQ_NONE && type_of(diff.add_pc) == PAWN)
        return true;  // (rare) a pawn was added
    return false;
}

void PawnStruct::append_changed_indices(
  Color, const Position& pos, const DiffType& diff, IndexList& removed, IndexList& added) {
    if (!pawn_structure_changed(diff))
        return;

    // NEW pawns (current position is post-move).
    Bitboard wpNew = pos.pieces(WHITE, PAWN), bpNew = pos.pieces(BLACK, PAWN);

    // Reconstruct OLD pawns by undoing the dirty-piece changes that touch pawns.
    Bitboard wpOld = wpNew, bpOld = bpNew;
    auto     ptr = [&](Color c) -> Bitboard& { return c == WHITE ? wpOld : bpOld; };

    if (type_of(diff.pc) == PAWN)
    {
        Color c = color_of(diff.pc);
        ptr(c) |= square_bb(diff.from);                 // pawn was at 'from'
        if (diff.to != SQ_NONE)
            ptr(c) &= ~square_bb(diff.to);              // and not yet at 'to' (non-promotion)
        // promotion: to==SQ_NONE, pawn left the board -> only the |from restore applies
    }
    if (diff.remove_sq != SQ_NONE && type_of(diff.remove_pc) == PAWN)
        ptr(color_of(diff.remove_pc)) |= square_bb(diff.remove_sq);  // captured pawn was present
    if (diff.add_sq != SQ_NONE && type_of(diff.add_pc) == PAWN)
        ptr(color_of(diff.add_pc)) &= ~square_bb(diff.add_sq);       // added pawn wasn't present

    // OLD indices -> removed, NEW indices -> added (only emit when they differ).
    int lo = left_index(wpOld, bpOld), ro = right_index(wpOld, bpOld);
    int ln = left_index(wpNew, bpNew), rn = right_index(wpNew, bpNew);
    if (lo != ln)
    {
        if (lo >= 0)
            removed.push_back(IndexType(lo));
        if (ln >= 0)
            added.push_back(IndexType(ln));
    }
    if (ro != rn)
    {
        if (ro >= 0)
            removed.push_back(IndexType(NumLeft + ro));
        if (rn >= 0)
            added.push_back(IndexType(NumLeft + rn));
    }
}

bool PawnStruct::requires_refresh(const DiffType&, Color) {
    return false;  // king-independent; refreshes are driven by the king-relative feature sets
}

}  // namespace Stockfish::Eval::NNUE::Features
