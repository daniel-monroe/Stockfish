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
struct DualNetworkOutput {
    Value psqt;
    Value mainPositional;
    Value overPositional;
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

    // Load only the layer-stack (FC) weights of a second, byte-identical-FT net
    // (e.g. nn-overestimate.nnue) into overestimateNetwork[]. The FT read from
    // the file is asserted equal to the already-loaded main FT, then discarded.
    // On any mismatch/failure the overestimate head is left disabled.
    void load_overestimate(const std::string& rootDirectory, std::string evalfilePath);

    std::size_t get_content_hash() const;

    NetworkOutput evaluate(const Position&    pos,
                           AccumulatorStack&  accumulatorStack,
                           AccumulatorCaches& cache) const;

    // Like evaluate(), but builds the accumulator once and runs the FC stack
    // twice: with the main FC weights and with the overestimate FC weights.
    // If the overestimate head is not loaded, overPositional == mainPositional.
    DualNetworkOutput evaluate_dual(const Position&    pos,
                                    AccumulatorStack&  accumulatorStack,
                                    AccumulatorCaches& cache) const;

    // True only when a distinct overestimate net has been loaded via
    // load_overestimate(). When false the overestimate head is a copy of the main
    // head (uncertainty == 0). The head itself is always initialized (from main).
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

    bool read_header(std::istream&, std::uint32_t*, std::string*) const;
    bool write_header(std::ostream&, std::uint32_t, const std::string&) const;

    bool read_parameters(std::istream&, std::string&);
    bool write_parameters(std::ostream&, const std::string&) const;

    // Input feature converter
    FeatureTransformer featureTransformer;

    // Evaluation function (main value head)
    NetworkArchitecture network[LayerStacks];

    // Second FC head (overestimate / uncertainty). Shares featureTransformer.
    // Only populated by load_overestimate(); guarded by overestimateLoaded.
    NetworkArchitecture overestimateNetwork[LayerStacks];

    EvalFile evalFile;

    bool initialized        = false;
    // overestimateLoaded: the overestimate FC head is initialized (always true after
    //   a main net load; it is a copy of the main head until a real net overwrites it).
    // overestimateIsReal: a distinct nn-overestimate.nnue has been loaded, so the head
    //   genuinely differs from main and uncertainty can be nonzero.
    bool overestimateLoaded = false;
    bool overestimateIsReal = false;

    // Hash value of evaluation function structure
    static constexpr std::uint32_t hash =
      FeatureTransformer::get_hash_value() ^ NetworkArchitecture::get_hash_value();

    friend struct AccumulatorCaches;
};


}  // namespace Stockfish

template<>
struct std::hash<Stockfish::Eval::NNUE::Network> {
    std::size_t operator()(const Stockfish::Eval::NNUE::Network& network) const noexcept {
        return network.get_content_hash();
    }
};

#endif
