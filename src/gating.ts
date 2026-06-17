"use strict";

/**
 * Gating match — head-to-head between two (policy, value) nets, each move by
 * greedy MCTS (temperature 0, no root noise). This replaces "material vs
 * random" as the promotion gate: a candidate replaces the champion only if it
 * scores above a margin against it. Far harder to saturate than beating a
 * random mover, and the standard AlphaZero promotion criterion.
 */

import { runMcts } from "@euriklis/mcts";
import { chessEnv, neuralEvaluator, seedRoot } from "./mcts-player";
import { type Color, type State, WHITE, startState, generateMoves, makeMove, inCheck, positionKey } from "./rules";
import { whiteMaterial, ConvValueNet } from "./value";
import { MLPPolicy } from "./policy";

export interface Net { policy: MLPPolicy; value: ConvValueNet; }
export interface MatchOpts { games?: number; numSimulations?: number; maxPlies?: number; adjudicateLead?: number; adjudicatePersist?: number; rng?: () => number; }

/** Play one game; return the winning colour (or 0 for draw). White/Black are nets. */
async function playGame(white: Net, black: Net, opts: Required<Omit<MatchOpts, "games">>): Promise<Color | 0> {
  const s: State = startState();
  let leadPlies = 0, leadColor: Color | 0 = 0;
  const repCount = new Map<string, number>();
  for (let ply = 0; ply < opts.maxPlies; ply++) {
    const moves = generateMoves(s);
    if (moves.length === 0) return inCheck(s) ? ((-s.turn) as Color) : 0;
    if (s.halfmove >= 100) return 0;
    const key = positionKey(s);
    const prior = repCount.get(key) ?? 0;
    repCount.set(key, prior + 1);
    if (prior + 1 >= 3) return 0; // threefold repetition
    seedRoot(s, repCount, prior);
    const side = s.turn === WHITE ? white : black;
    const res = await runMcts(chessEnv, neuralEvaluator(side.policy, side.value), s, {
      numSimulations: opts.numSimulations, cPuct: 1.5, temperature: 0, dirichletAlpha: 0, backup: "negamax", rng: opts.rng,
    });
    makeMove(s, res.action);
    const wm = whiteMaterial(s.board);
    const cur: Color | 0 = Math.abs(wm) >= opts.adjudicateLead ? (Math.sign(wm) as Color) : 0;
    if (cur !== 0 && cur === leadColor) leadPlies++;
    else { leadColor = cur; leadPlies = cur !== 0 ? 1 : 0; }
    if (leadPlies >= opts.adjudicatePersist) return leadColor as Color;
  }
  return 0;
}

export interface MatchResult { aWins: number; bWins: number; draws: number; aScore: number; games: number; }

/** Play `games` games (alternating colours) between A and B → A's score share. */
export async function playMatch(a: Net, b: Net, opts: MatchOpts = {}): Promise<MatchResult> {
  const games = opts.games ?? 20;
  const cfg = {
    numSimulations: opts.numSimulations ?? 100,
    maxPlies: opts.maxPlies ?? 160,
    adjudicateLead: opts.adjudicateLead ?? 6,
    adjudicatePersist: opts.adjudicatePersist ?? 6,
    rng: opts.rng ?? Math.random,
  };
  let aWins = 0, bWins = 0, draws = 0;
  for (let g = 0; g < games; g++) {
    const aIsWhite = g % 2 === 0;
    const winnerColor = aIsWhite ? await playGame(a, b, cfg) : await playGame(b, a, cfg);
    const aColor: Color = aIsWhite ? WHITE : (-WHITE as Color);
    if (winnerColor === 0) draws++;
    else if (winnerColor === aColor) aWins++;
    else bWins++;
  }
  return { aWins, bWins, draws, aScore: (aWins + 0.5 * draws) / games, games };
}
