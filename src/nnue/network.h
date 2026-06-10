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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "../types.h"
#include "../misc.h"
#include "nnue_architecture.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {

class AccumulatorStack;
struct AccumulatorCaches;

using NetworkOutput = std::tuple<Value, Value>;

// Output of a dual-head evaluation: the shared PSQT term, plus the positional
// value from the main FC head and from the overestimate FC head. All in the
// same internal units (pre-OutputScale-division). uncertainty = over - main.
//
// finalHidden additionally carries the 32 clipped fc_1 activations (uint8) that
// feed the main output layer fc_2 — i.e. the SAME final-hidden layer ("the 32
// neurons before the final projection") that the uncertainty head branches off.
// Plumbed out unconditionally and passively (it influences no search decision)
// so the search-side diagnostics can correlate each neuron with prune outcomes.
struct DualNetworkOutput {
    Value psqt;
    Value mainPositional;
    Value overPositional;
    std::array<std::uint8_t, NetworkArchitecture::UncertaintyInputDims> finalHidden;
};

// The network must be a trivial type, i.e. the memory must be in-line.
// This is required to allow sharing the network via shared memory, as
// there is no way to run destructors.
class Network {
   public:
    Network(EvalFile file) :
        evalFile(file) {}

    Network(const Network& other) = default;
    Network(Network&& other)      = default;

    Network& operator=(const Network& other) = default;
    Network& operator=(Network&& other)      = default;

    void load(const std::string& rootDirectory, std::string evalfilePath);
    bool save(const std::optional<std::string>& filename) const;

    // Load the single-linear uncertainty head from a standalone dual-head .nnue
    // file (one that carries the "OVRHEAD1" trailer). The main net payload is
    // read into temporaries solely to reach the trailer; only the trailer's
    // per-bucket linear head is kept. On any mismatch/failure the head is left
    // as-is (a no-op preserves whatever was loaded from the main EvalFile).
    void load_overestimate(const std::string& rootDirectory, std::string evalfilePath);

    usize get_content_hash() const;

    NetworkOutput evaluate(const Position&    pos,
                           AccumulatorStack&  accumulatorStack,
                           AccumulatorCaches& cache) const;

    // Like evaluate(), but builds the accumulator once, runs the single main FC
    // stack once, and additionally evaluates the cheap per-bucket single-linear
    // uncertainty head on the SAME feature-transformer output. The head produces
    // a scalar delta in the same integer units as the main head's "skip" term;
    // overPositional == mainPositional + delta. If the head is zero (no trailer),
    // delta == 0 and overPositional == mainPositional exactly.
    DualNetworkOutput evaluate_dual(const Position&    pos,
                                    AccumulatorStack&  accumulatorStack,
                                    AccumulatorCaches& cache) const;

    // True only when a real (nonzero) uncertainty head has been loaded, i.e. the
    // single file carried an "OVRHEAD1" trailer (or a head was loaded via
    // load_overestimate). When false the head is all-zeros so the delta — and
    // hence the uncertainty — is exactly 0.
    bool has_overestimate() const { return overestimateIsReal; }


    void verify(std::string evalfilePath, const std::function<void(std::string_view)>&) const;
    NnueEvalTrace trace_evaluate(const Position&    pos,
                                 AccumulatorStack&  accumulatorStack,
                                 AccumulatorCaches& cache) const;

   private:
    void load_user_net(const std::string&, const std::string&);
    void load_internal();

    void initialize();

    bool                       save(std::ostream&, const std::string&, const std::string&) const;
    std::optional<std::string> load(std::istream&);

    bool read_header(std::istream&, u32*, std::string*) const;
    bool write_header(std::ostream&, u32, const std::string&) const;

    bool read_parameters(std::istream&, std::string&);
    bool write_parameters(std::ostream&, const std::string&) const;

    // Helper: read the optional "OVRHEAD1" uncertainty-head trailer that may
    // follow the main net payload. Returns false only on a corrupt/short trailer
    // (a clean EOF with no magic is fine and leaves the head zeroed).
    bool read_uncertainty_trailer(std::istream& stream);

    // Read just the per-bucket single-linear head body (int32 bias + 32 int8
    // weights per bucket) into uncBias[]/uncWeights[]. Sets overestimateIsReal.
    bool read_uncertainty_head(std::istream& stream);

    // Width of the uncertainty head's input: the final-hidden activations that
    // feed the main output layer fc_2 (the trainer's l2x_), = FC_1_OUTPUTS = L3.
    static constexpr std::size_t UncDims = NetworkArchitecture::UncertaintyInputDims;

    // Input feature converter
    FeatureTransformer featureTransformer;

    // Evaluation function (main value head)
    NetworkArchitecture network[LayerStacks];

    // Uncertainty / "overestimate" head: ONE linear layer per layer-stack bucket
    // that branches off the main head's FINAL HIDDEN LAYER. It maps the clipped
    // fc_1 activations (the UncDims = FC_1_OUTPUTS = 32 uint8 values that feed the
    // main output layer fc_2, i.e. the trainer's l2x_) to a single scalar delta.
    // Quantized with the SAME weight scale as the main fc_2 (ls_output), so the
    // delta lives in the same integer units as the main head's pre-scaling output
    // (fc_2_out[0] + skip) and can be added straight onto it. Zeroed when no
    // trailer is present -> delta 0 -> overestimate == main exactly.
    alignas(CacheLineSize) std::int8_t uncWeights[LayerStacks][UncDims];
    alignas(CacheLineSize) std::int32_t uncBias[LayerStacks];

    EvalFile evalFile;

    bool initialized = false;
    // overestimateIsReal: a real (nonzero) uncertainty head is loaded, so the
    //   delta — and hence uncertainty — can be nonzero. When false the head is
    //   all-zeros and the delta is exactly 0 (overestimate == main).
    bool overestimateIsReal = false;

    // Hash value of evaluation function structure
    static constexpr u32 hash =
      FeatureTransformer::get_hash_value() ^ NetworkArchitecture::get_hash_value();

    friend struct AccumulatorCaches;
};


}  // namespace Stockfish

template<>
struct std::hash<Stockfish::Eval::NNUE::Network> {
    Stockfish::usize operator()(const Stockfish::Eval::NNUE::Network& network) const noexcept {
        return network.get_content_hash();
    }
};

#endif
