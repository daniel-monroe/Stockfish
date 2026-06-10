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

#include "network.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../incbin/incbin.h"

#include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_misc.h"
#include "nnz_helper.h"
#include "simd.h"

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(UNIVERSAL_BINARY) && !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#elif defined(UNIVERSAL_BINARY_MACOS_X86_SLICE)
// Determined at runtime, see universal/nnue_embed.cpp
extern const unsigned char* const gEmbeddedNNUEData;
extern const unsigned int         gEmbeddedNNUESize;
#elif defined(UNIVERSAL_BINARY)
extern const unsigned char gEmbeddedNNUEData[];
extern const unsigned int  gEmbeddedNNUESize;
#else
const unsigned char gEmbeddedNNUEData[1] = {0x0};
const unsigned int  gEmbeddedNNUESize    = 1;
#endif

namespace Stockfish::Eval::NNUE {


namespace Detail {

// Read evaluation function parameters
template<typename T>
bool read_parameters(std::istream& stream, T& reference) {

    u32 header;
    header = read_little_endian<u32>(stream);
    if (!stream || header != T::get_hash_value())
        return false;
    return reference.read_parameters(stream);
}

// Write evaluation function parameters
template<typename T>
bool write_parameters(std::ostream& stream, const T& reference) {

    write_little_endian<u32>(stream, T::get_hash_value());
    return reference.write_parameters(stream);
}

}  // namespace Detail

namespace {

// Width of the uncertainty head's input: the final-hidden activations that feed
// the main output layer fc_2 (the trainer's l2x_).
constexpr std::size_t UncDims = NetworkArchitecture::UncertaintyInputDims;  // = L3 = 32

// Single-linear uncertainty head: dot the UncDims (=32) uint8 final-hidden
// activations (the clipped fc_1 output that feeds the main output layer fc_2)
// with the int8 per-bucket uncertainty weights, plus the int32 bias. This is the
// same affine math as the main fc_2 (input uint8 unsigned, weights int8 signed),
// with a single output, quantized with the same ls_output weight scale — so the
// result is already in the same integer units as fc_2_out[0] / fwdOut.
//
// It is only 32 int8 MACs (one AVX2 dpbusd), so a tiny dense loop is plenty; no
// sparsity is involved (the input here is the dense 32-wide hidden layer, not the
// sparse feature-transformer output).
#if defined(USE_AVX2)
inline std::int32_t uncertainty_dot(const std::uint8_t* input,
                                    const std::int8_t*  weights,
                                    std::int32_t        bias) {
    static_assert(UncDims == sizeof(__m256i), "UncDims must be 32 for the AVX2 path");
    __m256i acc = _mm256_setzero_si256();
    SIMD::m256_add_dpbusd_epi32(acc, _mm256_load_si256(reinterpret_cast<const __m256i*>(input)),
                                _mm256_load_si256(reinterpret_cast<const __m256i*>(weights)));
    return SIMD::m256_hadd(acc, bias);
}
#else
inline std::int32_t uncertainty_dot(const std::uint8_t* input,
                                    const std::int8_t*  weights,
                                    std::int32_t        bias) {
    std::int32_t sum = bias;
    for (std::size_t i = 0; i < UncDims; ++i)
        sum += std::int32_t(input[i]) * std::int32_t(weights[i]);
    return sum;
}
#endif

}  // namespace

void Network::load(const std::string& rootDirectory, std::string evalfilePath) {
#if defined(DEFAULT_NNUE_DIRECTORY)
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory,
                                     stringify(DEFAULT_NNUE_DIRECTORY)};
#else
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory};
#endif

    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    for (const auto& directory : dirs)
    {
        if (std::string(evalFile.current) != evalfilePath)
        {
            if (directory != "<internal>")
                load_user_net(directory, evalfilePath);
            else if (evalfilePath == std::string(evalFile.defaultName))
                load_internal();
        }
    }
}


bool Network::save(const std::optional<std::string>& filename) const {
    std::string actualFilename;
    std::string msg;

    if (filename.has_value())
        actualFilename = filename.value();
    else
    {
        if (std::string(evalFile.current) != std::string(evalFile.defaultName))
        {
            msg = "Failed to export a net. "
                  "A non-embedded net can only be saved if the filename is specified";

            sync_cout << msg << sync_endl;
            return false;
        }

        actualFilename = evalFile.defaultName;
    }

    std::ofstream stream(actualFilename, std::ios_base::binary);
    bool          saved = save(stream, evalFile.current, evalFile.netDescription);

    msg = saved ? "Network saved successfully to " + actualFilename : "Failed to export a net";

    sync_cout << msg << sync_endl;
    return saved;
}


NetworkOutput Network::evaluate(const Position&    pos,
                                AccumulatorStack&  accumulatorStack,
                                AccumulatorCaches& cache) const {

    constexpr u64 alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NNZInfo<L1> nnzInfo;

    const int  bucket     = (pos.count<ALL_PIECES>() - 1) / 4;
    const auto psqt       = featureTransformer.transform(pos, accumulatorStack, cache,
                                                         transformedFeatures, bucket, nnzInfo);
    const auto positional = network[bucket].propagate(transformedFeatures, nnzInfo);
    return {static_cast<Value>(psqt / OutputScale), static_cast<Value>(positional / OutputScale)};
}


DualNetworkOutput Network::evaluate_dual(const Position&    pos,
                                         AccumulatorStack&  accumulatorStack,
                                         AccumulatorCaches& cache) const {

    constexpr uint64_t alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NNZInfo<L1> nnzInfo;

    const int  bucket = (pos.count<ALL_PIECES>() - 1) / 4;
    const auto psqt   = featureTransformer.transform(pos, accumulatorStack, cache,
                                                      transformedFeatures, bucket, nnzInfo);

    // Run the single main FC stack; the accumulator/FT is shared. We also get
    // back the raw pre-scaling value (fwdOut) so the uncertainty delta can be
    // added in the same integer units and re-scaled jointly. propagate() also
    // copies out the clipped fc_1 activations — the final-hidden layer (the input
    // to the main output layer fc_2, i.e. the trainer's l2x_) — that the
    // uncertainty head branches off.
    alignas(alignment) std::uint8_t fc2Input[UncDims];
    std::int32_t                    mainFwd        = 0;
    const auto mainPositional = network[bucket].propagate(transformedFeatures, nnzInfo, mainFwd,
                                                          fc2Input);

    // Uncertainty head: ONE linear layer on the final-hidden activations (the
    // UncDims=32 uint8 values that feed the main output layer fc_2), quantized
    // with the same ls_output weight scale as fc_2. The result is therefore in
    // the same integer units as fc_2's output / the main head's pre-scaling sum
    // (fwdOut), so we add it to fwdOut and run it through the identical final
    // scaling. Zero weights -> delta 0 -> overPositional == mainPositional.
    const std::int32_t deltaInt =
      uncertainty_dot(fc2Input, uncWeights[bucket], uncBias[bucket]);

    const auto overPositional =
      NetworkArchitecture::scale_fwd_out(std::int64_t(mainFwd) + std::int64_t(deltaInt));

    DualNetworkOutput out{static_cast<Value>(psqt / OutputScale),
                          static_cast<Value>(mainPositional / OutputScale),
                          static_cast<Value>(overPositional / OutputScale),
                          {}};
    // Carry out the 32 final-hidden activations (the input to fc_2) so search-side
    // diagnostics can correlate each neuron with prune outcomes. Passive: it feeds
    // no search decision. These are the SAME bytes the uncertainty head consumed.
    std::memcpy(out.finalHidden.data(), fc2Input, UncDims * sizeof(std::uint8_t));
    return out;
}


void Network::verify(std::string                                  evalfilePath,
                     const std::function<void(std::string_view)>& f) const {
    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (std::string(evalFile.current) != evalfilePath)
    {
        if (f)
        {
            std::string msg1 =
              "Network evaluation parameters compatible with the engine must be available.";
            std::string msg2 = "The network file " + evalfilePath + " was not loaded successfully.";
            std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                               "including the directory name, to the network file.";
            std::string msg4 = "The default net can be downloaded from: "
                               "https://tests.stockfishchess.org/api/nn/"
                             + std::string(evalFile.defaultName);
            std::string msg5 = "The engine will be terminated now.";

            std::string msg = "ERROR: " + msg1 + '\n' + "ERROR: " + msg2 + '\n' + "ERROR: " + msg3
                            + '\n' + "ERROR: " + msg4 + '\n' + "ERROR: " + msg5 + '\n';

            f(msg);
        }

        exit(EXIT_FAILURE);
    }

    if (f)
    {
        usize size = sizeof(featureTransformer) + sizeof(NetworkArchitecture) * LayerStacks;
        f("NNUE evaluation using " + evalfilePath + " (" + std::to_string(size / (1024 * 1024))
          + "MiB, (" + std::to_string(featureTransformer.InputDimensions) + ", "
          + std::to_string(network[0].TransformedFeatureDimensions) + ", "
          + std::to_string(network[0].FC_0_OUTPUTS) + ", " + std::to_string(network[0].FC_1_OUTPUTS)
          + ", 1))");
    }
}


NnueEvalTrace Network::trace_evaluate(const Position&    pos,
                                      AccumulatorStack&  accumulatorStack,
                                      AccumulatorCaches& cache) const {

    constexpr u64 alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NnueEvalTrace t{};
    t.correctBucket = (pos.count<ALL_PIECES>() - 1) / 4;
    for (IndexType bucket = 0; bucket < LayerStacks; ++bucket)
    {
        NNZInfo<L1> nnzInfo;
        const auto  materialist = featureTransformer.transform(pos, accumulatorStack, cache,
                                                               transformedFeatures, bucket, nnzInfo);
        const auto  positional  = network[bucket].propagate(transformedFeatures, nnzInfo);

        t.psqt[bucket]       = static_cast<Value>(materialist / OutputScale);
        t.positional[bucket] = static_cast<Value>(positional / OutputScale);
    }

    return t;
}


void Network::load_user_net(const std::string& dir, const std::string& evalfilePath) {
    std::ifstream stream(dir + evalfilePath, std::ios::binary);
    auto          description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalfilePath;
        evalFile.netDescription = description.value();
    }
}


void Network::load_overestimate(const std::string& rootDirectory, std::string evalfilePath) {
    // The uncertainty head now lives INSIDE the main .nnue file (the "OVRHEAD1"
    // trailer read in read_parameters), so this separate override is optional and
    // largely vestigial. An empty path PRESERVES whatever head the main EvalFile
    // already provided (zeroed for a plain net, real for a dual-head net).
    if (evalfilePath.empty())
        return;

    // Try the candidate directories in the same way load() does.
#if defined(DEFAULT_NNUE_DIRECTORY)
    std::vector<std::string> dirs = {"", rootDirectory, stringify(DEFAULT_NNUE_DIRECTORY)};
#else
    std::vector<std::string> dirs = {"", rootDirectory};
#endif

    std::ifstream stream;
    for (const auto& directory : dirs)
    {
        stream.open(directory + evalfilePath, std::ios::binary);
        if (stream.is_open())
            break;
        stream.clear();
    }
    if (!stream.is_open())
    {
        std::cerr << "Overestimate net not found: " << evalfilePath << std::endl;
        return;
    }

    // The override file is a full dual-head .nnue: header + FT + main stacks +
    // "OVRHEAD1" trailer. Read the main payload into temporaries (validated by
    // hash, but otherwise discarded) just to position the stream at the trailer.
    std::uint32_t hashValue;
    std::string   description;
    if (!read_header(stream, &hashValue, &description) || hashValue != Network::hash)
    {
        std::cerr << "Overestimate net header/arch mismatch: " << evalfilePath << std::endl;
        return;
    }

    auto tempFt = std::make_unique<FeatureTransformer>();
    if (!Detail::read_parameters(stream, *tempFt))
    {
        std::cerr << "Overestimate net: failed to read feature transformer." << std::endl;
        return;
    }

    auto tempStacks = std::make_unique<NetworkArchitecture[]>(LayerStacks);
    for (std::size_t i = 0; i < LayerStacks; ++i)
        if (!Detail::read_parameters(stream, tempStacks[i]))
        {
            std::cerr << "Overestimate net: failed to read layer stack " << i << std::endl;
            return;
        }

    // Now read the "OVRHEAD1" trailer + single-linear head into a temporary copy
    // so a failed/partial read does not clobber the head already in place.
    char magic[8] = {};
    stream.read(magic, sizeof(magic));
    static constexpr char UncertaintyTrailerMagic[8] = {'O', 'V', 'R', 'H', 'E', 'A', 'D', '1'};
    if (stream.gcount() != std::streamsize(sizeof(magic))
        || std::memcmp(magic, UncertaintyTrailerMagic, sizeof(magic)) != 0)
    {
        std::cerr << "Overestimate net: no OVRHEAD1 trailer; head unchanged." << std::endl;
        return;
    }

    std::int32_t tmpBias[LayerStacks];
    std::int8_t  tmpWeights[LayerStacks][UncDims];
    for (std::size_t b = 0; b < LayerStacks; ++b)
    {
        tmpBias[b] = read_little_endian<std::int32_t>(stream);
        for (std::size_t i = 0; i < UncDims; ++i)
            tmpWeights[b][i] = read_little_endian<std::int8_t>(stream);
    }
    if (!(stream && stream.peek() == std::ios::traits_type::eof()))
    {
        std::cerr << "Overestimate net: corrupt/short trailer; head unchanged." << std::endl;
        return;
    }

    std::memcpy(uncBias, tmpBias, sizeof(uncBias));
    std::memcpy(uncWeights, tmpWeights, sizeof(uncWeights));
    overestimateIsReal = true;
    std::cerr << "Overestimate head loaded: " << evalfilePath << std::endl;
}


void Network::load_internal() {
    // C++ way to prepare a buffer for a memory stream
    class MemoryBuffer: public std::basic_streambuf<char> {
       public:
        MemoryBuffer(char* p, usize n) {
            setg(p, p, p + n);
            setp(p, p + n);
        }
    };

#ifdef UNIVERSAL_BINARY_MACOS_X86_SLICE
    if (gEmbeddedNNUEData == nullptr)  // failed embedded load
        return;
#endif

    MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(gEmbeddedNNUEData)),
                        usize(gEmbeddedNNUESize));

    std::istream stream(&buffer);
    auto         description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalFile.defaultName;
        evalFile.netDescription = description.value();
    }
}


void Network::initialize() { initialized = true; }


bool Network::save(std::ostream&      stream,
                   const std::string& name,
                   const std::string& netDescription) const {
    if (name.empty() || name == "None")
        return false;

    return write_parameters(stream, netDescription);
}


std::optional<std::string> Network::load(std::istream& stream) {
    initialize();
    std::string description;

    return read_parameters(stream, description) ? std::make_optional(description) : std::nullopt;
}


usize Network::get_content_hash() const {
    if (!initialized)
        return 0;

    usize h = 0;
    hash_combine(h, featureTransformer);
    for (auto&& layerstack : network)
        hash_combine(h, layerstack);
    hash_combine(h, evalFile);
    return h;
}

// Read network header
bool Network::read_header(std::istream& stream, u32* hashValue, std::string* desc) const {
    u32 version, size;

    version    = read_little_endian<u32>(stream);
    *hashValue = read_little_endian<u32>(stream);
    size       = read_little_endian<u32>(stream);
    if (!stream || version != Version)
        return false;
    desc->resize(size);
    stream.read(&(*desc)[0], size);
    return !stream.fail();
}


// Write network header
bool Network::write_header(std::ostream& stream, u32 hashValue, const std::string& desc) const {
    write_little_endian<u32>(stream, Version);
    write_little_endian<u32>(stream, hashValue);
    write_little_endian<u32>(stream, u32(desc.size()));
    stream.write(&desc[0], desc.size());
    return !stream.fail();
}


bool Network::read_parameters(std::istream& stream, std::string& netDescription) {
    u32 hashValue;
    if (!read_header(stream, &hashValue, &netDescription))
        return false;
    if (hashValue != Network::hash)
        return false;
    if (!Detail::read_parameters(stream, featureTransformer))
        return false;
    for (usize i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::read_parameters(stream, network[i]))
            return false;
    }

    // GUARANTEE: the uncertainty head starts out all-zeros, so the delta — and
    // hence the uncertainty — is exactly 0 (overestimate == main) for a plain
    // single-head net. This runs per Network object, so under NUMA replication
    // each replica is independently (re)initialized here before replication.
    std::memset(uncWeights, 0, sizeof(uncWeights));
    std::memset(uncBias, 0, sizeof(uncBias));
    overestimateIsReal = false;

    // OPTIONAL SINGLE-FILE DUAL HEAD: the trunk (feature transformer) above is
    // shared; an 8-byte "OVRHEAD1" magic may be followed by the single-linear
    // uncertainty head (per bucket: int32 bias + 32 int8 weights). A plain net
    // without the trailer is fine (clean EOF) and leaves the head zeroed.
    return read_uncertainty_trailer(stream);
}


// Reads the optional "OVRHEAD1" trailer. Returns false only on a corrupt or
// short trailer; a clean EOF with no magic leaves the head zeroed and succeeds.
bool Network::read_uncertainty_trailer(std::istream& stream) {
    char magic[8] = {};
    stream.read(magic, sizeof(magic));
    const std::streamsize got = stream.gcount();

    static constexpr char UncertaintyTrailerMagic[8] = {'O', 'V', 'R', 'H', 'E', 'A', 'D', '1'};
    if (got == std::streamsize(sizeof(magic))
        && std::memcmp(magic, UncertaintyTrailerMagic, sizeof(magic)) == 0)
        return read_uncertainty_head(stream)
            && stream && stream.peek() == std::ios::traits_type::eof();

    // No trailer. A plain single-head net ends exactly here (clean EOF); any
    // partial read is unexpected trailing data.
    if (got != 0)
        return false;
    stream.clear();
    return true;
}


// Reads the per-bucket single-linear uncertainty head body, matching the
// trainer's NNUEWriter.write_fc_layer encoding for a (UncDims -> 1) layer:
//   per bucket: int32 bias (1 value), then UncDims (=FC_1_OUTPUTS=32) int8
//   weights (no padding, since 32 % 32 == 0). No per-bucket hash prefix.
bool Network::read_uncertainty_head(std::istream& stream) {
    for (std::size_t b = 0; b < LayerStacks; ++b)
    {
        uncBias[b] = read_little_endian<std::int32_t>(stream);
        for (std::size_t i = 0; i < UncDims; ++i)
            uncWeights[b][i] = read_little_endian<std::int8_t>(stream);
        if (!stream)
            return false;
    }
    overestimateIsReal = true;
    return true;
}


bool Network::write_parameters(std::ostream& stream, const std::string& netDescription) const {
    if (!write_header(stream, Network::hash, netDescription))
        return false;
    if (!Detail::write_parameters(stream, featureTransformer))
        return false;
    for (usize i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::write_parameters(stream, network[i]))
            return false;
    }
    return bool(stream);
}

}  // namespace Stockfish::Eval::NNUE
