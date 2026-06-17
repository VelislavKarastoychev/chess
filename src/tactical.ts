"use strict";

/**
 * Tactical test suite — a small FIXED thermometer that doesn't saturate the way
 * "score vs random" does. Each position has a forced mate-in-1; the engine
 * (policy priors + MCTS) should find a mating move. The expected set is
 * computed from the rules (every legal move that delivers checkmate), so the
 * positions can't silently rot — an invalid FEN with no mate-in-1 is flagged.
 *
 * As the value net and search improve, more of these pass with fewer
 * simulations. A handful of clear positions is enough to tell whether a change
 * helped or just moved noise around.
 */

import { type State, parseFEN, generateMoves, makeMove, unmakeMove, gameStatus, moveToUci } from "./rules";
import { chooseMoveMcts } from "./mcts-player";
import type { MLPPolicy } from "./policy";
import type { ConvValueNet } from "./value";

/** Mate-in-1 positions (White to move). */
const MATE_IN_1: string[] = [
  "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",        // Ra8#
  "7k/5ppp/8/8/8/8/8/R6K w - - 0 1",         // Ra8#
  "k7/8/1K6/8/8/8/8/7R w - - 0 1",           // Rh8#
  "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1",   // Qd8#
];

/** Every legal move that gives immediate checkmate (the accepted answers). */
function matingMoves(s: State): string[] {
  const out: string[] = [];
  for (const m of generateMoves(s)) {
    const u = makeMove(s, m);
    if (gameStatus(s) === "checkmate") out.push(moveToUci(m));
    unmakeMove(s, m, u);
  }
  return out;
}

export interface TacticalResult { passed: number; total: number; invalid: number; details: { fen: string; expected: string[]; got: string; ok: boolean }[]; }

export async function tacticalSuite(
  policy: MLPPolicy, value: ConvValueNet, opts: { numSimulations?: number } = {},
): Promise<TacticalResult> {
  const sims = opts.numSimulations ?? 120;
  const details: TacticalResult["details"] = [];
  let passed = 0, invalid = 0;
  for (const fen of MATE_IN_1) {
    const s = parseFEN(fen);
    const expected = matingMoves(s);
    if (expected.length === 0) { invalid++; details.push({ fen, expected, got: "—", ok: false }); continue; }
    const { move } = await chooseMoveMcts(parseFEN(fen), policy, value, { numSimulations: sims, temperature: 0 });
    const got = moveToUci(move);
    const ok = expected.includes(got);
    if (ok) passed++;
    details.push({ fen, expected, got, ok });
  }
  return { passed, total: MATE_IN_1.length, invalid, details };
}
