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

#include "tt.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "memory.h"
#include "misc.h"
#include "syzygy/tbprobe.h"
#include "thread.h"

namespace Stockfish {


// TTEntry struct is the 10 bytes transposition table entry, defined as:
//
// key        16 bit
// depth       8 bit
// pv node     1 bit
// bound type  2 bit
// generation  5 bit
// move       16 bit
// value      16 bit
// evaluation 16 bit
//
// These fields are in the same order as accessed by TT::probe(), since memory is fastest sequentially.
// Equally, the store order in save() matches this order.
//
// We use `bool(depth8)` as the cheap internal occupancy check, corresponding to `depth == DEPTH_NONE`
// externally, so we offset the internal depth by DEPTH_NONE.
//
// Pv, bound and generation are packed in a single byte.
//
// The NNUE overestimate-head futSignal (= overestimate_value - main_value, a
// non-negative internal-unit value, or VALUE_NONE = "not computed") is persisted
// WITHOUT growing the entry: it is quantized to a 5-bit code and packed into the
// cluster's 2 otherwise-unused padding bytes (3 entries * 5 bits = 15 of 16 bits).
// The entry therefore stays 10 bytes and the cluster stays 32 bytes, byte-identical
// to upstream, so this auxiliary signal does not perturb TT geometry, replacement,
// or search. It is NOT read by any pruning/search decision (see search.cpp).
static constexpr u8 GENERATION_BITS = 5;
static constexpr u8 GENERATION_MASK = (1 << GENERATION_BITS) - 1;
static constexpr u8 BOUND_SHIFT     = GENERATION_BITS;
static constexpr u8 BOUND_MASK      = 0b11 << BOUND_SHIFT;
static constexpr u8 PV_SHIFT        = BOUND_SHIFT + 2;
static constexpr u8 PV_MASK         = 1 << PV_SHIFT;

// 5-bit quantization for the packed-in-padding futSignal code. Uncertainty is
// either VALUE_NONE ("not computed for this node") or the SIGNED futility-head
// signal (deltaInt = acts.w8, pre-scaling integer units). Code 0 is reserved for
// VALUE_NONE; codes 1..31 encode 31 evenly-spaced levels spanning the signal's
// observed range [UNCERTAINTY_OFFSET .. UNCERTAINTY_OFFSET + 30*UNCERTAINTY_SCALE],
// i.e. about [-32000 .. +16000]; values outside saturate.
static constexpr int UNCERTAINTY_SCALE   = 497;     // internal units per quantization step
static constexpr int UNCERTAINTY_OFFSET  = -9443;   // value encoded by the lowest code
// (calibrated from the realistic-data deltaInt distribution of the fc2in futility
//  head, p0.5/p99.5; ~0.5% saturation each side; trigger gain survives quantization)
static constexpr int UNCERTAINTY_MAXLVL  = 30;      // codes 1..31 -> levels 0..30
static constexpr u16 UNCERTAINTY_BITS    = 5;
static constexpr u16 UNCERTAINTY_FIELD   = (1u << UNCERTAINTY_BITS) - 1;  // 0x1F

static u8 quantize_futSignal(Value futSignal) {
    if (futSignal == VALUE_NONE)
        return 0;  // sentinel: not computed
    int lvl = (int(futSignal) - UNCERTAINTY_OFFSET + UNCERTAINTY_SCALE / 2) / UNCERTAINTY_SCALE;
    if (lvl < 0)
        lvl = 0;
    if (lvl > UNCERTAINTY_MAXLVL)
        lvl = UNCERTAINTY_MAXLVL;
    return u8(lvl + 1);  // codes 1..31
}

static Value dequantize_futSignal(u8 code) {
    if (code == 0)
        return VALUE_NONE;
    return Value(UNCERTAINTY_OFFSET + (int(code) - 1) * UNCERTAINTY_SCALE);
}

// Extract / insert the 5-bit code for cluster slot `slot` (0..ClusterSize-1) within
// the cluster's packed 16-bit padding word.
static u8 unpack_futSignal(u16 packed, int slot) {
    return u8((packed >> (UNCERTAINTY_BITS * slot)) & UNCERTAINTY_FIELD);
}

static u16 pack_futSignal(u16 packed, int slot, u8 code) {
    const u16 sh = u16(UNCERTAINTY_BITS * slot);
    return u16((packed & ~(UNCERTAINTY_FIELD << sh)) | (u16(code & UNCERTAINTY_FIELD) << sh));
}

struct TTEntry {

    // Convert internal bitfields to external types. The futSignal is stored in the
    // cluster (not the entry), so the caller (probe) passes in the dequantized value.
    TTData read(Value futSignal) const {
        return TTData{Move(move16),
                      Value(value16),
                      Value(eval16),
                      Depth(DEPTH_NONE + depth8),
                      Bound((genBound8 & BOUND_MASK) >> BOUND_SHIFT),
                      bool(genBound8 & PV_MASK),
                      futSignal};
    }

    bool is_occupied() const { return bool(depth8); };
    // `clusterUnc`/`slot` locate this entry's 5-bit futSignal code in the parent
    // cluster's packed padding word; written only when the entry is overwritten.
    void save(Key                 k,
              Value               v,
              bool                pv,
              Bound               b,
              Depth               d,
              Move                m,
              Value               ev,
              Value               futSignal,
              u8                  curr_generation,
              RelaxedAtomic<u16>* clusterUnc,
              int                 slot);
    u8   relative_age(const u8 curr_generation) const;

   private:
    friend class TranspositionTable;
    friend struct TTWriter;

    RelaxedAtomic<u16>  key16;
    RelaxedAtomic<u8>   depth8;
    RelaxedAtomic<u8>   genBound8;
    RelaxedAtomic<Move> move16;
    RelaxedAtomic<i16>  value16;
    RelaxedAtomic<i16>  eval16;
};

// Populates the TTEntry with a new node's data, possibly
// overwriting an old position. The update is non-atomic and can be racy.
void TTEntry::save(Key                 k,
                   Value               v,
                   bool                pv,
                   Bound               b,
                   Depth               d,
                   Move                m,
                   Value               ev,
                   Value               futSignal,
                   u8                  curr_generation,
                   RelaxedAtomic<u16>* clusterUnc,
                   int                 slot) {

    // Preserve the old ttmove if we don't have a new one
    if (m || u16(k) != key16)
        move16 = m;

    // Overwrite less valuable entries (cheapest checks first)
    if (b == BOUND_EXACT || u16(k) != key16 || d - DEPTH_NONE + 2 * pv > depth8 - 4
        || relative_age(curr_generation))
    {
        assert(d > DEPTH_NONE);
        assert(d - DEPTH_NONE < 256);
        assert(curr_generation <= GENERATION_MASK);  // TT::new_search() plays nice

        key16     = u16(k);
        depth8    = u8(d - DEPTH_NONE);
        genBound8 = u8(curr_generation | b << BOUND_SHIFT | u8(pv) << PV_SHIFT);
        value16   = i16(v);
        eval16    = i16(ev);
        // Persist the futSignal alongside this entry, in the cluster's packed word.
        *clusterUnc = pack_futSignal(u16(*clusterUnc), slot, quantize_futSignal(futSignal));
    }
    // Secondary aging. Important for elementary mate finding.
    // (*Scaler) Secondary aging on entries relevant to singular extensions
    // generally scales poorly and requires VVLTC verification.
    else if (depth8 + DEPTH_NONE >= 5
             && Bound((genBound8 & BOUND_MASK) >> BOUND_SHIFT) != BOUND_EXACT)
    {
        auto v16 = value16;
        if (std::abs(v16) < VALUE_INFINITE && is_decisive(v16))
            depth8 = std::max(int(depth8) - 1,
                              0);  // guard against racy underflows, default to "unoccupied"
    }
}


u8 TTEntry::relative_age(const u8 curr_generation) const {
    // Returns this entry's age. We count generations like clocks count hours,
    // i.e. we require 0 - 1 == 31. Unsigned subtraction guarantees the required
    // borrowing regardless of the upper pv/bound bits.
    return (curr_generation - genBound8) & GENERATION_MASK;
}


// TTWriter is but a very thin wrapper around the pointer
TTWriter::TTWriter(TTEntry* tte, RelaxedAtomic<u16>* unc, int s) :
    entry(tte),
    clusterUnc(unc),
    slot(s) {}

void TTWriter::write(Key   k,
                     Value v,
                     bool  pv,
                     Bound b,
                     Depth d,
                     Move  m,
                     Value ev,
                     Value futSignal,
                     u8    curr_generation) {
    entry->save(k, v, pv, b, d, m, ev, futSignal, curr_generation, clusterUnc, slot);
}

void TTWriter::penalize(int penalty) {
    // guard against racy underflows, default to "unoccupied"
    entry->depth8 = std::max(int(entry->depth8) - penalty, 0);
}


// A TranspositionTable is an array of Cluster, of size clusterCount. Each cluster consists of ClusterSize number
// of TTEntry. Each non-empty TTEntry contains information on exactly one position. The size of a Cluster should
// divide the size of a cache line for best performance, as the cacheline is prefetched when possible.

static constexpr int ClusterSize = 3;

// The 2 bytes that upstream leaves as dead padding (3 * 10-byte entries = 30, padded
// to 32) are repurposed to hold the three entries' 5-bit futSignal codes (15 of 16
// bits). The entry stays 10 bytes and the cluster stays 32 bytes, byte-identical to
// upstream's geometry.
struct Cluster {
    TTEntry            entry[ClusterSize];
    RelaxedAtomic<u16> futSignal;  // 3 packed 5-bit codes; was `char padding[2]`
};

static_assert(sizeof(TTEntry) == 10, "Unexpected TTEntry size");
static_assert(sizeof(Cluster) == 32, "Suboptimal Cluster size");


// Sets the size of the transposition table,
// measured in megabytes. Transposition table consists
// of clusters and each cluster consists of ClusterSize number of TTEntry.
void TranspositionTable::resize(usize mbSize, ThreadPool& threads) {
    aligned_large_pages_free(table);

    clusterCount  = mbSize * 1024 * 1024 / sizeof(Cluster);
    usize ttBytes = clusterCount * sizeof(Cluster);

    // Request 1GB pages if we'd get at least eight per NUMA node, to avoid
    // memory oversubscription
    bool hugePageHint = ttBytes >= threads.numa_nodes() * HugePageSize * 8;

    table = static_cast<Cluster*>(aligned_large_pages_alloc_with_hint(ttBytes, hugePageHint));

    if (!table)
    {
        std::cerr << "Failed to allocate " << mbSize << "MB for transposition table." << std::endl;
        exit(EXIT_FAILURE);
    }

    clear(threads);
}


// Initializes the entire transposition table to zero,
// in a multi-threaded way.
void TranspositionTable::clear(ThreadPool& threads) {
    generation8             = 0;
    const usize threadCount = threads.num_threads();

    std::vector<usize> threadToNuma = threads.get_bound_thread_to_numa_node();

    std::vector<usize> order(threadCount);
    std::iota(order.begin(), order.end(), 0);

    // To promote good NUMA distribution (esp. with huge pages), we permute threads so that
    // all threads in a NUMA node clear a contiguous region of the TT.
    if (threadToNuma.size() == threadCount)
    {
        std::stable_sort(order.begin(), order.end(), [&threadToNuma](usize t1, usize t2) {
            return threadToNuma.at(t1) < threadToNuma.at(t2);
        });
    }

    for (usize i = 0; i < threadCount; ++i)
    {
        threads.run_on_thread(order[i], [this, i, threadCount]() {
            // Each thread will zero its part of the hash table
            const usize stride = clusterCount / threadCount;
            const usize start  = stride * i;
            const usize len    = i + 1 != threadCount ? stride : clusterCount - start;

            std::memset(static_cast<void*>(&table[start]), 0, len * sizeof(Cluster));
        });
    }

    for (usize i = 0; i < threadCount; ++i)
        threads.wait_on_thread(i);
}


// Returns an approximation of the hashtable
// occupation during a search. The hash is x permill full, as per UCI protocol.
// Only counts entries which are younger than maxAge.
int TranspositionTable::hashfull(int maxAge) const {
    int cnt = 0;
    for (int i = 0; i < 1000; ++i)
        for (int j = 0; j < ClusterSize; ++j)
            cnt += table[i].entry[j].is_occupied()
                && table[i].entry[j].relative_age(generation8) <= maxAge;

    return cnt / ClusterSize;
}


void TranspositionTable::new_search() {
    ++generation8;
    // Don't overflow into the other bits of TTEntry::genBound8
    generation8 &= GENERATION_MASK;
}


u8 TranspositionTable::generation() const { return generation8; }


// Looks up the current position in the transposition table.
// It returns true if the key is found (which may be a collision), and has non-null data.
// Otherwise, it returns false and a pointer to an empty or least valuable TTEntry
// to be replaced later. The value of an entry is its depth minus 8 times its relative age.
std::tuple<bool, TTData, TTWriter> TranspositionTable::probe(const Key key) const {

    TTEntry* const tte   = first_entry(key);
    const u16      key16 = u16(key);  // Use the low 16 bits as key inside the cluster
    // tte points at entry[0], the first member of its Cluster, so the cluster (and its
    // packed futSignal word) is recoverable by reinterpreting that pointer.
    Cluster* const cluster = reinterpret_cast<Cluster*>(tte);

    for (int i = 0; i < ClusterSize; ++i)
        if (tte[i].key16 == key16)
            // This gap is the main place for read races.
            // After `read()` completes that copy is final, but may be self-inconsistent.
            return {tte[i].is_occupied(),
                    tte[i].read(dequantize_futSignal(
                      unpack_futSignal(u16(cluster->futSignal), i))),
                    TTWriter(&tte[i], &cluster->futSignal, i)};

    // Find an entry to be replaced according to the replacement strategy
    TTEntry* replace   = tte;
    int      replaceIdx = 0;
    for (int i = 1; i < ClusterSize; ++i)
        if (replace->depth8 - 8 * replace->relative_age(generation8)
            > tte[i].depth8 - 8 * tte[i].relative_age(generation8))
        {
            replace    = &tte[i];
            replaceIdx = i;
        }

    return {false,
            TTData{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false, VALUE_NONE},
            TTWriter(replace, &cluster->futSignal, replaceIdx)};
}


TTEntry* TranspositionTable::first_entry(const Key key) const {
    return &table[mul_hi64(key, clusterCount)].entry[0];
}

}  // namespace Stockfish
