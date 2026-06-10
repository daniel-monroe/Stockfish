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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
namespace {

// Shared scaling: turn the raw NNUE (psqt, positional) outputs into the final
// static evaluation. Factored out so the plain and dual-head entry points
// produce a byte-identical static eval.
Value scale_nnue_eval(Value psqt, Value positional, int optimism, const Position& pos) {

    Value nnue = (125 * psqt + 131 * positional) / 128;

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = std::abs(psqt - positional);
    optimism += optimism * nnueComplexity / 476;
    nnue -= nnue * nnueComplexity / 18236;

    int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
    int v        = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 199;

    // Guarantee evaluation does not hit the tablebase range
    return std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

}  // namespace

Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());

    auto [psqt, positional] = network.evaluate(pos, accumulators, caches);
    return scale_nnue_eval(psqt, positional, optimism, pos);
}

// Single-pass variant: one feature-transformer/body evaluation, both FC heads.
// Returns the same static eval as evaluate() and also the uncertainty.
Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism,
                     Value&                         uncertainty,
                     std::uint8_t*                  finalHidden) {

    assert(!pos.checkers());

    const auto dual = network.evaluate_dual(pos, accumulators, caches);

    // Diagnostic plumbing: hand the 32 pre-fc_2 activations back to the caller.
    if (finalHidden != nullptr)
        std::memcpy(finalHidden, dual.finalHidden.data(), dual.finalHidden.size());

    // Run BOTH heads through the identical raw -> internal-units transformation,
    // so the uncertainty is expressed in the same (final static eval) space that
    // search uses, not raw network units. Only the body (feature transformer) is
    // shared/expensive; this second scale is cheap scalar arithmetic.
    const Value mainV = scale_nnue_eval(dual.psqt, dual.mainPositional, optimism, pos);
    const Value overV = scale_nnue_eval(dual.psqt, dual.overPositional, optimism, pos);

    uncertainty = overV - mainV;
    return mainV;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Network& network) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(network);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, network, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    auto  dual          = network.evaluate_dual(pos, *accumulators, *caches);
    Value mainValue     = dual.psqt + dual.mainPositional;
    Value overValue     = dual.psqt + dual.overPositional;
    Value v             = mainValue;
    ss << "NNUE evaluation          " << v << " (side to move, internal units)\n";
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    // Auxiliary overestimate / uncertainty head (search uses main value only).
    // The uncertainty is reported in the same (scaled internal-units) space that
    // search stores, i.e. both heads run through the shared scale_nnue_eval.
    {
        Value mainW = pos.side_to_move() == WHITE ? mainValue : -mainValue;
        Value overW = pos.side_to_move() == WHITE ? overValue : -overValue;
        Value mainScaled = scale_nnue_eval(dual.psqt, dual.mainPositional, VALUE_ZERO, pos);
        Value overScaled = scale_nnue_eval(dual.psqt, dual.overPositional, VALUE_ZERO, pos);
        Value uncertainty = overScaled - mainScaled;
        ss << "NNUE main eval         " << 0.01 * UCIEngine::to_cp(mainW, pos)
           << " (white side)\n";
        ss << "NNUE overestimate eval " << 0.01 * UCIEngine::to_cp(overW, pos) << " (white side)"
           << (network.has_overestimate() ? "" : " [overestimate net not loaded]") << '\n';
        ss << "NNUE uncertainty       " << 0.01 * UCIEngine::to_cp(uncertainty, pos)
           << " (over - main, scaled)\n";
    }

    v = evaluate(network, pos, *accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;

    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
