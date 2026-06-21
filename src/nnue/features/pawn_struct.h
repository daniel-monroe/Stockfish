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

// Definition of input feature PawnStruct of NNUE evaluation function.
//
// Two per-half-board pawn-structure feature blocks (left = files a-d, right =
// files e-h) concatenated into one feature set. Each block maps the position's
// color-folded (flip-side canonical) per-side pawn configuration to its rank in
// a frequency-ordered vocabulary (pawn_vocab.h), firing exactly one feature per
// block. The feature is perspective-symmetric (the canonical key is identical
// from both perspectives), so make_index ignores the perspective. Out-of-vocab
// structures fire nothing.
//
// NumLeft / NumRight MUST match the trained net (the trainer feature string
// "PawnStructLeft:NumLeft+PawnStructRight:NumRight").

#ifndef NNUE_FEATURES_PAWN_STRUCT_H_INCLUDED
#define NNUE_FEATURES_PAWN_STRUCT_H_INCLUDED

#include <cstdint>

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class PawnStruct {
   public:
    // Vocabulary sizes per side -- MUST equal the trained net's PawnStructLeft:N / Right:N.
    static constexpr IndexType NumLeft  = 4096;
    static constexpr IndexType NumRight = 4096;

    static constexpr Bitboard LeftMask  = 0x0F0F0F0F0F0F0F0FULL;
    static constexpr Bitboard RightMask = 0xF0F0F0F0F0F0F0F0ULL;

    // Hash value embedded in the evaluation file. Folds both sides' base hashes
    // with their vocab sizes, matching the trainer's per-component HASH
    // (PawnStruct._BASE_HASH ^ num_inputs) combined left-then-right.
    static constexpr u32 BaseLeft  = 0x5A1F0001u;
    static constexpr u32 BaseRight = 0x5A1F0002u;
    static constexpr u32 HashLeft  = BaseLeft ^ NumLeft;
    static constexpr u32 HashRight = BaseRight ^ NumRight;
    // combine_hash(left, right): rotate-left-1 then xor (see ComposedFeatureTransformer._compute_hash)
    static constexpr u32 HashValue =
      (((HashLeft << 1) | (HashLeft >> 31)) ^ HashRight) & 0xFFFFFFFFu;

    // Number of feature dimensions (left block then right block).
    static constexpr IndexType Dimensions = NumLeft + NumRight;

    // At most one feature per side fires.
    static constexpr IndexType MaxActiveDimensions = 2;
    using IndexList                                = ValueList<IndexType, MaxActiveDimensions>;
    using DiffType                                 = DirtyPiece;

    // Get a list of indices for active features (perspective-independent).
    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    // Get a list of indices for recently changed features.
    static void append_changed_indices(
      Color perspective, const Position& pos, const DiffType& diff, IndexList& removed, IndexList& added);

    // The pawn feature is king-independent; it never forces a full refresh on its own.
    static bool requires_refresh(const DiffType& diff, Color perspective);

    // True if the dirty piece set changed pawn placement (pawn moved/captured/promoted).
    static bool pawn_structure_changed(const DiffType& diff);

   private:
    // Look up the vocab rank of one half-board's canonical pawn structure; -1 if OOV.
    static int  left_index(Bitboard wp, Bitboard bp);
    static int  right_index(Bitboard wp, Bitboard bp);
    static void indices_for(const Position& pos, IndexList& out);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_PAWN_STRUCT_H_INCLUDED
