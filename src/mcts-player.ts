"use strict";

/**
 * Chess adapter for the generic @euriklis/mcts engine: wraps the trained
 * policy/value nets as an AlphaZero-style PUCT search. The network supplies
 * priors (which moves to explore) and leaf values (how good a position is);
 * MCTS does the lookahead that the raw network can't — catching hanging pieces,
 * finding mates, and avoiding stalemates the policy alone walked into.
 */

import { runMcts, type MctsEnv, type Evaluator, type MctsResult } from "@euriklis/mcts";
import {
  type State, type Move,
  generateMoves, makeMove, inCheck,
} from "./rules";
import { featureMatrix } from "./features";
import { positionFeatures, ValueNet } from "./value";
import { MLPPolicy } from "./policy";

const cloneState = (s: State): State => ({
  board: Int8Array.from(s.board), turn: s.turn, castling: s.castling,
  ep: s.ep, halfmove: s.halfmove, fullmove: s.fullmove,
});

/** The chess game as an MctsEnv (zero-sum, terminal = mate/stalemate/50-move). */
export const chessEnv: MctsEnv<State, Move> = {
  legalActions: (s) => generateMoves(s),
  apply: (s, m) => { const c = cloneState(s); makeMove(c, m); return c; },
  isTerminal: (s) => generateMoves(s).length === 0 || s.halfmove >= 100,
  reward: (s) => (generateMoves(s).length === 0 ? (inCheck(s) ? -1 : 0) : 0), // side to move mated ⇒ −1
};

/** Neural leaf evaluator: policy → move priors, value net → position value. */
export const neuralEvaluator = (policy: MLPPolicy, value: ValueNet): Evaluator<State, Move> =>
  async (s, legal) => {
    const out = await policy.forward(featureMatrix(s, legal));
    const v = await value.forward(positionFeatures(s));
    return { value: (v.view as Float64Array)[0]!, priors: Array.from(out.probs.view as Float64Array) };
  };

export interface MctsPlayOpts {
  numSimulations?: number;
  cPuct?: number;
  temperature?: number;     // 0 = strongest (argmax visits); >0 = varied
  dirichletAlpha?: number;  // root noise (use for self-play training, not for play)
  rng?: () => number;
}

/** Pick a move by MCTS over the trained nets. Returns the move and search stats. */
export async function chooseMoveMcts(
  state: State, policy: MLPPolicy, value: ValueNet, opts: MctsPlayOpts = {},
): Promise<{ move: Move; search: MctsResult<Move> }> {
  const search = await runMcts(chessEnv, neuralEvaluator(policy, value), state, {
    numSimulations: opts.numSimulations ?? 200,
    cPuct: opts.cPuct ?? 1.5,
    temperature: opts.temperature ?? 0,
    dirichletAlpha: opts.dirichletAlpha ?? 0,
    backup: "negamax",
    rng: opts.rng,
  });
  return { move: search.action, search };
}
