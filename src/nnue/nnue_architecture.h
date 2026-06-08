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

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <cstdint>
#include <cstring>
#include <iosfwd>

#include "features/half_ka_v2_hm.h"
#include "features/full_threats.h"
#include "layers/affine_transform.h"
#include "layers/affine_transform_sparse_input.h"
#include "layers/clipped_relu.h"
#include "layers/sqr_clipped_relu.h"
#include "nnue_common.h"
#include "nnz_helper.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using ThreatFeatureSet = Features::FullThreats;
using PSQFeatureSet    = Features::HalfKAv2_hm;

// Number of input feature dimensions after conversion
constexpr IndexType L1 = 1024;
constexpr int       L2 = 31;
constexpr int       L3 = 32;

constexpr IndexType PSQTBuckets = 8;
constexpr IndexType LayerStacks = 8;

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
static_assert(PSQTBuckets % 8 == 0,
              "Per feature PSQT values cannot be processed at granularity lower than 8 at a time.");

struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;
    static constexpr int       FC_1_OUTPUTS                 = L3;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS + 1> fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS + 1, WeightScaleBits + 1>                      ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS + 1, WeightScaleBits + 1>                         ac_0;
    Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>                            fc_1;
    Layers::ClippedReLU<FC_1_OUTPUTS, WeightScaleBits>                                 ac_1;
    Layers::AffineTransform<FC_1_OUTPUTS, 1>                                           fc_2;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
        // input slice hash
        std::uint32_t hashValue = 0xEC42E90Du;
        hashValue ^= TransformedFeatureDimensions * 2;

        hashValue = decltype(fc_0)::get_hash_value(hashValue);
        // TODO: considerincluding hash value of ac_sqr_0 in the overall hash value.
        // For now omitted on purpose because hash value is not written by trainer yet
        hashValue = decltype(ac_0)::get_hash_value(hashValue);
        hashValue = decltype(fc_1)::get_hash_value(hashValue);
        hashValue = decltype(ac_1)::get_hash_value(hashValue);
        hashValue = decltype(fc_2)::get_hash_value(hashValue);

        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        return fc_0.read_parameters(stream) && ac_0.read_parameters(stream)
            && fc_1.read_parameters(stream) && ac_1.read_parameters(stream)
            && fc_2.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        return fc_0.write_parameters(stream) && ac_0.write_parameters(stream)
            && fc_1.write_parameters(stream) && ac_1.write_parameters(stream)
            && fc_2.write_parameters(stream);
    }

    std::int32_t propagate(const TransformedFeatureType* transformedFeatures,
                           const NNZInfo<L1>&            nnzInfo) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
            alignas(CacheLineSize) typename decltype(ac_sqr_0)::OutputType
              ac_sqr_0_out[ceil_to_multiple<IndexType>(FC_0_OUTPUTS * 2, 32)];
            alignas(CacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
            alignas(CacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;

            Buffer() { std::memset(ac_sqr_0_out, 0, sizeof(ac_sqr_0_out)); }
        };

        Buffer buffer;

        fc_0.propagate(transformedFeatures, buffer.fc_0_out, nnzInfo);
        ac_sqr_0.propagate(buffer.fc_0_out, buffer.ac_sqr_0_out);
        ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
        std::memcpy(buffer.ac_sqr_0_out + FC_0_OUTPUTS, buffer.ac_0_out,
                    FC_0_OUTPUTS * sizeof(typename decltype(ac_0)::OutputType));
        fc_1.propagate(buffer.ac_sqr_0_out, buffer.fc_1_out);
        ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);
        fc_2.propagate(buffer.ac_1_out, buffer.fc_2_out);

        // max value for fwdOut is (L1 + L3) * HiddenMaxVal * WeightMaxVal
        // for int8 activations and weights this is (L1 + L3) * 16129 making
        // fwdOut safe from overflow until (L1 + L3) > 133,144
        // first layer and last layer use WeightScaleBits + 1
        std::int32_t fwdOut = buffer.fc_2_out[0] + buffer.fc_0_out[FC_0_OUTPUTS];
        // fwdOut is such that 1.0 is equal to HiddenOneVal*(1<<WeightScaleBits)*2 in
        // quantized form, but we want 1.0 to be equal to 600*OutputScale
        // to make overflow impossible we cast to int64_t
        constexpr std::int64_t multiplier  = 600 * OutputScale;
        constexpr std::int64_t denominator = static_cast<std::int64_t>(HiddenOneVal)
                                           * static_cast<std::int64_t>(1U << WeightScaleBits) * 2;

        std::int32_t outputValue =
          static_cast<std::int32_t>((static_cast<std::int64_t>(fwdOut) * multiplier) / denominator);
        return outputValue;
    }

    // Applies the SAME final scaling propagate() uses to map an internal
    // accumulator value (in "fwdOut" units, where the main head's skip term and
    // the uncertainty head's delta both live) into the engine's pre-OutputScale
    // value units. Used by the uncertainty head so its delta rides the exact
    // same quantization path as the main output.
    static std::int32_t scale_fwd_out(std::int64_t fwdOut) {
        constexpr std::int64_t multiplier  = 600 * OutputScale;
        constexpr std::int64_t denominator = static_cast<std::int64_t>(HiddenOneVal)
                                           * static_cast<std::int64_t>(1U << WeightScaleBits) * 2;
        return static_cast<std::int32_t>((fwdOut * multiplier) / denominator);
    }

    // Number of activations that feed the output layer fc_2 (the "final hidden
    // layer" the uncertainty head branches off, matching the trainer's l2x_).
    static constexpr int UncertaintyInputDims = FC_1_OUTPUTS;  // = L3 = 32

    // Like propagate(), but also returns the raw pre-scaling accumulator value
    // (fwdOut), so a caller can add an extra term (the uncertainty delta) in the
    // same units and re-scale jointly for exact parity with the trainer. It also
    // copies out the clipped fc_1 activations (the input to the output layer
    // fc_2) — the SAME final-hidden activations the trainer's uncertainty head
    // consumes — so the caller can run the single-linear uncertainty head on them.
    std::int32_t propagate(const TransformedFeatureType* transformedFeatures,
                           const NNZInfo<L1>&            nnzInfo,
                           std::int32_t&                 fwdOutRaw,
                           std::uint8_t fc2Input[UncertaintyInputDims]) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
            alignas(CacheLineSize) typename decltype(ac_sqr_0)::OutputType
              ac_sqr_0_out[ceil_to_multiple<IndexType>(FC_0_OUTPUTS * 2, 32)];
            alignas(CacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
            alignas(CacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;

            Buffer() { std::memset(ac_sqr_0_out, 0, sizeof(ac_sqr_0_out)); }
        };

        Buffer buffer;

        fc_0.propagate(transformedFeatures, buffer.fc_0_out, nnzInfo);
        ac_sqr_0.propagate(buffer.fc_0_out, buffer.ac_sqr_0_out);
        ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
        std::memcpy(buffer.ac_sqr_0_out + FC_0_OUTPUTS, buffer.ac_0_out,
                    FC_0_OUTPUTS * sizeof(typename decltype(ac_0)::OutputType));
        fc_1.propagate(buffer.ac_sqr_0_out, buffer.fc_1_out);
        ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);

        // ac_1_out is the clipped fc_1 output: the uint8 input to fc_2 (the
        // output layer), i.e. the trainer's final-hidden activations l2x_. Hand
        // these to the uncertainty head. ac_1's OutputBuffer is uint8 and its
        // logical width is FC_1_OUTPUTS.
        std::memcpy(fc2Input, buffer.ac_1_out, UncertaintyInputDims * sizeof(std::uint8_t));

        fc_2.propagate(buffer.ac_1_out, buffer.fc_2_out);

        fwdOutRaw = buffer.fc_2_out[0] + buffer.fc_0_out[FC_0_OUTPUTS];
        return scale_fwd_out(fwdOutRaw);
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, fc_0.get_content_hash());
        hash_combine(h, ac_sqr_0.get_content_hash());
        hash_combine(h, ac_0.get_content_hash());
        hash_combine(h, fc_1.get_content_hash());
        hash_combine(h, ac_1.get_content_hash());
        hash_combine(h, fc_2.get_content_hash());
        hash_combine(h, get_hash_value());
        return h;
    }
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::NetworkArchitecture> {
    std::size_t operator()(const Stockfish::Eval::NNUE::NetworkArchitecture& arch) const noexcept {
        return arch.get_content_hash();
    }
};

#endif  // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
