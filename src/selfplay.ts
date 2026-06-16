"use strict";

/**
 * Step 4 — reinforcement learning by self-play (REINFORCE with a value
 * baseline = actor-critic).
 *
 *   π(a|s)  scores legal moves (policy.ts).
 *   V(s)    predicts the return (value.ts), the variance-reducing baseline.
 *   loss  = −(1/N)Σ Aₜ·log π(aₜ|sₜ)  +  c·(1/N)Σ (V(sₜ) − Gₜ)²
 *           with advantage Aₜ = Gₜ − V(sₜ) treated as a constant for the policy.
 *
 * Cold-start chess RL from terminal reward alone is mostly draws → no signal,
 * exactly the trap flagged in the design. So returns are densified with
 * *material shaping*: each ply contributes the discounted material it swings
 * (mover's perspective), plus a terminal win/loss bonus. The policy gets a
 * gradient every move — learn to win material and not hang pieces — long before
 * it can force mate.
 *
 * `trainStep` is generic over a list of transitions, so the mechanics can be
 * unit-tested on a hand-built bandit (tests/reinforce.test.ts).
 */

import { Tensor } from "@euriklis/mathematics/tensor";
import {
  type Color, type Move, type State,
  startState, generateMoves, makeMove, inCheck,
} from "./rules";
import { featureMatrix } from "./features";
import { positionFeatures, whiteMaterial, ValueNet } from "./value";
import { MLPPolicy, type Policy } from "./policy";

// ---- small seeded RNG so runs are reproducible ------------------------------
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const sampleFromProbs = (probs: number[], rng: () => number): number => {
  let r = rng(), acc = 0;
  for (let i = 0; i < probs.length; i++) { acc += probs[i]!; if (r <= acc) return i; }
  return probs.length - 1;
};
const argmax = (xs: number[]): number => {
  let bi = 0; for (let i = 1; i < xs.length; i++) if (xs[i]! > xs[bi]!) bi = i; return bi;
};

// ---- tensor scalar helpers --------------------------------------------------
const scalar = (c: number): Tensor => { const t = new Tensor({ shape: [1, 1], type: "float64" }); t.view[0] = c; return t; };
const rowOneHot = (L: number, idx: number): Tensor => {
  const t = new Tensor({ shape: [1, L], type: "float64" }); t.view[idx] = 1; return t;
};
const rowOnes = (L: number): Tensor => {
  const t = new Tensor({ shape: [1, L], type: "float64" }); t.view.fill(1); return t;
};
const sumTensors = async (ts: Tensor[]): Promise<Tensor> => {
  let acc = ts[0]!;
  for (let i = 1; i < ts.length; i++) acc = await acc.plus(ts[i]!);
  return acc;
};

// ---- a single recorded decision --------------------------------------------
export interface Transition {
  moveFeats: number[][];   // [L, F] candidate-move features
  posFeat: number[];       // position features for the value net
  chosenIdx: number;       // which move was played
  G: number;               // return (filled by computeReturns)
  mover: Color;
}

export interface RolloutOpts { maxPlies?: number; gamma?: number; pawnScale?: number; winScale?: number; }

/** Play one self-play game; record every decision (both colours share π). */
export async function selfPlayGame(policy: Policy, rng: () => number, opts: RolloutOpts = {}) {
  const maxPlies = opts.maxPlies ?? 60;
  const s = startState();
  const decisions: Transition[] = [];
  const bw: number[] = [whiteMaterial(s.board)];
  let result: Color | 0 = 0;

  for (let ply = 0; ply < maxPlies; ply++) {
    const moves = generateMoves(s);
    if (moves.length === 0) { result = inCheck(s) ? ((-s.turn) as Color) : 0; break; }
    if (s.halfmove >= 100) { result = 0; break; }
    const feats = featureMatrix(s, moves);
    const out = await policy.forward(feats);
    const probs = Array.from(out.probs.view as Float64Array);
    const idx = sampleFromProbs(probs, rng);
    decisions.push({ moveFeats: feats, posFeat: positionFeatures(s), chosenIdx: idx, mover: s.turn, G: 0 });
    makeMove(s, moves[idx]!);
    bw.push(whiteMaterial(s.board));
  }
  computeReturns(decisions, bw, result, opts);
  return { decisions, result, plies: decisions.length };
}

/** Discounted return per decision: material swing (mover view) + terminal bonus. */
export function computeReturns(decisions: Transition[], bw: number[], result: Color | 0, opts: RolloutOpts) {
  const gamma = opts.gamma ?? 0.98, pawn = opts.pawnScale ?? 0.1, win = opts.winScale ?? 1.0;
  const T = decisions.length;
  for (let p = 0; p < T; p++) {
    const c = decisions[p]!.mover;
    let G = 0, disc = 1;
    for (let k = p + 1; k <= T; k++) {
      let r = c * (bw[k]! - bw[k - 1]!) * pawn;          // material gained at ply k, mover's view
      if (k === T && result !== 0) r += c * result * win; // terminal win/loss
      G += disc * r; disc *= gamma;
    }
    decisions[p]!.G = G;
  }
}

export interface TrainOpts { valueCoef?: number; advBaseline?: "value" | "mean" | "none"; entropyCoef?: number; }

/**
 * One optimiser step over a batch of transitions. `value` may be null (then the
 * baseline is the batch-mean return or zero). Returns the loss components.
 */
export async function trainStep(
  policy: Policy, value: ValueNet | null, transitions: Transition[],
  opt: { zeroGrad: () => void; step: () => Promise<void> }, opts: TrainOpts = {},
): Promise<{ loss: number; policyLoss: number; valueLoss: number }> {
  const valueCoef = opts.valueCoef ?? 0.5;
  const entropyCoef = opts.entropyCoef ?? 0;
  const baseline = opts.advBaseline ?? (value ? "value" : "mean");
  const meanG = transitions.reduce((a, t) => a + t.G, 0) / Math.max(1, transitions.length);

  const polTerms: Tensor[] = [], valTerms: Tensor[] = [], entTerms: Tensor[] = [];
  for (const t of transitions) {
    const L = t.moveFeats.length;
    const out = await policy.forward(t.moveFeats);
    const probA = await rowOneHot(L, t.chosenIdx).times(out.probs); // [1,1]
    const logpA = await probA.log();

    let v: Tensor | null = null, baseVal = 0;
    if (value) { v = await value.forward(t.posFeat); baseVal = (v.view as Float64Array)[0]!; }
    else if (baseline === "mean") baseVal = meanG;

    const adv = t.G - baseVal;                    // detached scalar coefficient
    polTerms.push(await logpA.hadamard(scalar(-adv)));
    if (v) { const d = await v.minus(scalar(t.G)); valTerms.push(await d.hadamard(d)); }
    if (entropyCoef > 0) {
      // Σₐ pₐ·log pₐ  (= −entropy). Minimising +coef·this raises entropy.
      const plogp = await out.probs.hadamard(await out.probs.log());   // [L,1]
      entTerms.push(await rowOnes(L).times(plogp));                    // [1,1]
    }
  }

  const invN = scalar(1 / polTerms.length);
  const policyLoss = await (await sumTensors(polTerms)).hadamard(invN);
  let loss = policyLoss, valueLoss: Tensor | null = null;
  if (valTerms.length) {
    valueLoss = await (await sumTensors(valTerms)).hadamard(invN);
    loss = await policyLoss.plus(await valueLoss.hadamard(scalar(valueCoef)));
  }
  if (entTerms.length) {
    const entLoss = await (await sumTensors(entTerms)).hadamard(invN); // mean Σ p log p (≤0)
    loss = await loss.plus(await entLoss.hadamard(scalar(entropyCoef)));
  }

  opt.zeroGrad();
  await loss.backward();
  await opt.step();
  return {
    loss: (loss.view as Float64Array)[0]!,
    policyLoss: (policyLoss.view as Float64Array)[0]!,
    valueLoss: valueLoss ? (valueLoss.view as Float64Array)[0]! : 0,
  };
}

/** Greedy evaluation vs a uniform-random opponent. Returns score & material. */
export async function evaluateVsRandom(policy: Policy, rng: () => number, nGames = 20, maxPlies = 80) {
  let score = 0, material = 0;
  for (let g = 0; g < nGames; g++) {
    const ourColor: Color = g % 2 === 0 ? 1 : -1;
    const s = startState();
    let result: Color | 0 = 0;
    for (let ply = 0; ply < maxPlies; ply++) {
      const moves = generateMoves(s);
      if (moves.length === 0) { result = inCheck(s) ? ((-s.turn) as Color) : 0; break; }
      if (s.halfmove >= 100) break;
      let idx: number;
      if (s.turn === ourColor) {
        const out = await policy.forward(featureMatrix(s, moves));
        idx = argmax(Array.from(out.probs.view as Float64Array));
      } else idx = Math.floor(rng() * moves.length);
      makeMove(s, moves[idx]!);
    }
    score += result === 0 ? 0.5 : result === ourColor ? 1 : 0;
    material += ourColor * whiteMaterial(s.board);
  }
  return { score: score / nGames, material: material / nGames };
}

// ===========================================================================
// Opponent-league training (v2): play OUR side against an opponent drawn from a
// pool (random + frozen snapshots of past policies). Avoids the mirror-self-play
// passivity collapse, and shapes rewards to punish stalemating a won position.
// ===========================================================================

export type Opponent = (s: State, moves: Move[]) => Promise<number> | number;

export const randomOpponent = (rng: () => number): Opponent =>
  (_s, moves) => Math.floor(rng() * moves.length);

/** Frozen policy opponent (temperature-sampled). */
export const policyOpponent = (p: MLPPolicy, rng: () => number, temp = 0.5): Opponent =>
  async (s, moves) => {
    const out = await p.forward(featureMatrix(s, moves));
    const logits = Array.from(out.scores.view as Float64Array).map((x) => x / temp);
    const m = Math.max(...logits);
    const ex = logits.map((x) => Math.exp(x - m));
    const z = ex.reduce((a, b) => a + b, 0);
    return sampleFromProbs(ex.map((e) => e / z), rng);
  };

/** Deep-copy a policy's weights into a new frozen instance (for the league). */
export function clonePolicy(p: MLPPolicy): MLPPolicy {
  const c = new MLPPolicy({ hidden: p.hidden });
  p.parameters().forEach((src, i) => c.parameters()[i]!.view.set(Array.from(src.view)));
  return c;
}

const posKey = (s: State): string => `${s.board.join(",")}|${s.turn}|${s.castling}|${s.ep}`;

export interface VsOpts extends RolloutOpts {
  drawPenalty?: number;  // flat penalty for any draw (decisiveness)
  mateBonus?: number;    // extra reward on checkmate (on top of winScale)
  repPenalty?: number;   // penalty for repeating a position (anti-shuffle)
}

/** Play one game: our policy as `ourColor`, opponent from the pool. Records our moves. */
export async function playVsOpponent(
  policy: Policy, opponent: Opponent, ourColor: Color, rng: () => number, opts: VsOpts = {},
): Promise<{ transitions: Transition[]; result: Color | 0; terminal: string; plies: number }> {
  const maxPlies = opts.maxPlies ?? 100;
  const gamma = opts.gamma ?? 0.98, pawn = opts.pawnScale ?? 0.1, win = opts.winScale ?? 1.0;
  const drawPen = opts.drawPenalty ?? 0.3, mateBonus = opts.mateBonus ?? 1.5, repPen = opts.repPenalty ?? 0.15;

  const s = startState();
  const bw: number[] = [whiteMaterial(s.board)];
  const repAt: number[] = [];                  // per-ply repetition penalty
  const ours: { moveFeats: number[][]; posFeat: number[]; chosenIdx: number; ply: number }[] = [];
  const seen = new Map<string, number>([[posKey(s), 1]]);
  let result: Color | 0 = 0, terminal = "cap";

  for (let ply = 0; ply < maxPlies; ply++) {
    const moves = generateMoves(s);
    if (moves.length === 0) { result = inCheck(s) ? ((-s.turn) as Color) : 0; terminal = inCheck(s) ? "checkmate" : "stalemate"; break; }
    if (s.halfmove >= 100) { result = 0; terminal = "draw50"; break; }

    let idx: number;
    if (s.turn === ourColor) {
      const feats = featureMatrix(s, moves);
      const out = await policy.forward(feats);
      idx = sampleFromProbs(Array.from(out.probs.view as Float64Array), rng);
      ours.push({ moveFeats: feats, posFeat: positionFeatures(s), chosenIdx: idx, ply });
    } else {
      idx = await opponent(s, moves);
    }
    makeMove(s, moves[idx]!);
    bw.push(whiteMaterial(s.board));
    const k = (seen.get(posKey(s)) ?? 0) + 1; seen.set(posKey(s), k);
    repAt[ply] = k >= 2 ? repPen : 0;          // revisiting a position is discouraged
  }

  // Terminal reward from our perspective. Stalemating/drawing a WON position hurts.
  const T = bw.length - 1;
  const ourFinalMat = ourColor * bw[T]!;
  let termR = 0;
  if (terminal === "checkmate") termR = (result === ourColor ? 1 : -1) * (win + mateBonus);
  else if (terminal !== "cap") termR = -drawPen - Math.max(0, ourFinalMat) * pawn * 0.5;

  const transitions: Transition[] = ours.map((d) => {
    let G = 0, disc = 1;
    for (let k = d.ply + 1; k <= T; k++) {
      let r = ourColor * (bw[k]! - bw[k - 1]!) * pawn - (repAt[k - 1] ?? 0);
      if (k === T) r += termR;
      G += disc * r; disc *= gamma;
    }
    return { moveFeats: d.moveFeats, posFeat: d.posFeat, chosenIdx: d.chosenIdx, mover: ourColor, G };
  });
  return { transitions, result, terminal, plies: ours.length };
}
