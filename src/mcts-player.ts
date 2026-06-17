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
  generateMoves, makeMove, inCheck, positionKey,
} from "./rules";
import { featureMatrix } from "./features";
import { ConvValueNet } from "./value";
import { MLPPolicy } from "./policy";

const cloneState = (s: State): State => ({
  board: Int8Array.from(s.board), turn: s.turn, castling: s.castling,
  ep: s.ep, halfmove: s.halfmove, fullmove: s.fullmove,
});

// ── Repetition tracking through the search ──────────────────────────────────
// The generic engine threads only the state, so we carry each search state's
// repetition context in a side WeakMap (keyed by the state object — distinct
// per node, GC'd when the search ends, safe across concurrent games). `base` is
// the game-history position counts (shared, read-only during a search), `path`
// the position keys visited from the root, `reps` the prior occurrences of THIS
// position (base + path). Lets isTerminal see threefold draws inside the search
// and the value evaluator pass the real repetition count to the planes.
type RepInfo = { reps: number; base: Map<string, number>; path: string[] };
const repInfo = new WeakMap<State, RepInfo>();
const EMPTY_BASE: Map<string, number> = new Map();

/** Seed the search root with the game's repetition history (`base` = counts of
 * positions before the root; `rootReps` = prior occurrences of the root). */
export const seedRoot = (root: State, base: Map<string, number>, rootReps: number): void => {
  repInfo.set(root, { reps: rootReps, base, path: [] });
};
/** Prior-occurrence count of a search state (0 if untracked, e.g. plain play). */
export const repsOf = (s: State): number => repInfo.get(s)?.reps ?? 0;

/** The chess game as an MctsEnv — terminal = mate / stalemate / 50-move /
 *  threefold repetition (the last detected via the repInfo side-table). */
export const chessEnv: MctsEnv<State, Move> = {
  legalActions: (s) => generateMoves(s),
  apply: (s, m) => {
    const c = cloneState(s);
    makeMove(c, m);
    const info = repInfo.get(s);
    const base = info?.base ?? EMPTY_BASE;
    const path = info?.path ?? [];
    const key = positionKey(c);
    let pc = 0;
    for (const k of path) if (k === key) pc++;
    repInfo.set(c, { reps: (base.get(key) ?? 0) + pc, base, path: [...path, key] });
    return c;
  },
  isTerminal: (s) => generateMoves(s).length === 0 || s.halfmove >= 100 || repsOf(s) >= 2,
  reward: (s) => (generateMoves(s).length === 0 ? (inCheck(s) ? -1 : 0) : 0), // mate ⇒ −1; repetition/50-move ⇒ 0
};

/** Neural leaf evaluator: policy → move priors, conv value net → position value
 *  (with the leaf's repetition count fed to the value planes). */
export const neuralEvaluator = (policy: MLPPolicy, value: ConvValueNet): Evaluator<State, Move> =>
  async (s, legal) => {
    const out = await policy.forward(featureMatrix(s, legal));
    const v = await value.forward(s, repsOf(s));
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
  state: State, policy: MLPPolicy, value: ConvValueNet, opts: MctsPlayOpts = {},
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
