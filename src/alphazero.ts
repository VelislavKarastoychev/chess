"use strict";

/**
 * AlphaZero-style training (stage 2): self-play where every move is chosen by
 * MCTS, and the network learns to imitate the SEARCH, not its own raw moves.
 *
 *   policy target π = MCTS visit distribution at the root (search-improved)
 *   value  target z = REAL game outcome (mover's perspective): natural mate,
 *                     or adjudicated win once a side holds a decisive material
 *                     lead — so the value signal is a genuine ±1, not a
 *                     material proxy. Only truly unresolved games fall back to
 *                     a soft material tanh.
 *
 *   loss = mean( −Σₐ πₐ·log p(a|s) )   +   c · mean( (V(s) − z)² )
 *
 * Self-play runs many games CONCURRENTLY through a BatchedEvaluator, so the
 * conv value net amortises over a batch of leaves per forward — that's what
 * makes a deeper search affordable. Samples accumulate in a replay buffer and
 * each step trains on a sampled minibatch (lower variance, less forgetting).
 */

import { Tensor } from "@euriklis/mathematics/tensor";
import { runMcts, type Evaluator } from "@euriklis/mcts";
import { chessEnv } from "./mcts-player";
import { type Color, type State, type Move, startState, generateMoves, makeMove, inCheck } from "./rules";
import { featureMatrix } from "./features";
import { whiteMaterial, ConvValueNet } from "./value";
import { PLANE_DIM, encodePlanes } from "./planes";
import { MLPPolicy } from "./policy";
import { BatchedEvaluator } from "./batched-eval";

const scalar = (c: number): Tensor => { const t = new Tensor({ shape: [1, 1], type: "float64" }); t.view[0] = c; return t; };
const rowVec = (xs: number[]): Tensor => { const t = new Tensor({ shape: [1, xs.length], type: "float64" }); t.view.set(xs); return t; };
const sumT = async (ts: Tensor[]): Promise<Tensor> => { let a = ts[0]!; for (let i = 1; i < ts.length; i++) a = await a.plus(ts[i]!); return a; };

export interface AZSample { moveFeats: number[][]; planes: Float64Array; pi: number[]; z: number; }

export interface AZRolloutOpts {
  numSimulations?: number; cPuct?: number; maxPlies?: number;
  dirichletAlpha?: number; materialScale?: number;
  adjudicateLead?: number;    // |material| ≥ this for `adjudicatePersist` plies ⇒ decisive
  adjudicatePersist?: number;
}

/** One MCTS self-play game through a shared (batched) evaluator → samples. */
export async function selfPlayGame(
  evaluator: Evaluator<State, Move>, rng: () => number, opts: AZRolloutOpts = {},
): Promise<{ samples: AZSample[]; result: Color | 0; terminal: string; plies: number }> {
  const sims = opts.numSimulations ?? 100, maxPlies = opts.maxPlies ?? 160;
  const cPuct = opts.cPuct ?? 1.5, dir = opts.dirichletAlpha ?? 0.3, matScale = opts.materialScale ?? 0.2;
  const adjLead = opts.adjudicateLead ?? 6, adjPersist = opts.adjudicatePersist ?? 6;

  const s = startState();
  const recs: { moveFeats: number[][]; planes: Float64Array; pi: number[]; mover: Color }[] = [];
  let result: Color | 0 = 0, terminal = "truncated";
  let leadPlies = 0, leadColor: Color | 0 = 0;

  for (let ply = 0; ply < maxPlies; ply++) {
    const moves = generateMoves(s);
    if (moves.length === 0) { result = inCheck(s) ? ((-s.turn) as Color) : 0; terminal = inCheck(s) ? "checkmate" : "stalemate"; break; }
    if (s.halfmove >= 100) { result = 0; terminal = "draw50"; break; }

    const res = await runMcts(chessEnv, evaluator, s, {
      numSimulations: sims, cPuct, temperature: 1, dirichletAlpha: dir, backup: "negamax", rng,
    });
    recs.push({ moveFeats: featureMatrix(s, moves), planes: encodePlanes(s), pi: res.policy, mover: s.turn });
    makeMove(s, res.action);

    // Adjudication: a sustained decisive material lead counts as a win, so the
    // value head sees real ±1 outcomes instead of endless truncated games.
    const wm = whiteMaterial(s.board);
    const cur: Color | 0 = Math.abs(wm) >= adjLead ? (Math.sign(wm) as Color) : 0;
    if (cur !== 0 && cur === leadColor) leadPlies++;
    else { leadColor = cur; leadPlies = cur !== 0 ? 1 : 0; }
    if (leadPlies >= adjPersist) { result = leadColor as Color; terminal = "adjudicated"; break; }
  }

  const finalMat = whiteMaterial(s.board);
  const decisive = terminal === "checkmate" || terminal === "adjudicated";
  const samples: AZSample[] = recs.map((r) => ({
    moveFeats: r.moveFeats, planes: r.planes, pi: r.pi,
    z: decisive ? (result === r.mover ? 1 : -1) : Math.tanh(r.mover * finalMat * matScale),
  }));
  return { samples, result, terminal, plies: recs.length };
}

export interface SelfPlayBatchResult {
  samples: AZSample[];
  decisive: number;       // games ending in mate/adjudication
  games: number;
  avgPlies: number;
  batches: number;        // evaluator forward batches
  avgBatch: number;       // mean positions per forward (batching efficiency)
}

/** Run `games` self-play games CONCURRENTLY, coalescing leaf evals into batches. */
export async function selfPlayBatch(
  policy: MLPPolicy, value: ConvValueNet, rng: () => number,
  opts: AZRolloutOpts & { games?: number; maxBatch?: number } = {},
): Promise<SelfPlayBatchResult> {
  const games = opts.games ?? 16;
  const ev = new BatchedEvaluator(policy, value, opts.maxBatch ?? 256);
  ev.active = games;
  const evaluate = ev.evaluator();

  // Per-game deterministic rng stream derived from the base rng.
  const streams = Array.from({ length: games }, (_, g) => {
    let a = (1000003 * (g + 1)) ^ Math.floor(rng() * 0x7fffffff);
    return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  });

  let done = 0;
  const results = await Promise.all(streams.map((r) =>
    selfPlayGame(evaluate, r, opts).finally(() => { done++; ev.active = games - done; }),
  ));

  const samples: AZSample[] = [];
  let decisive = 0, plies = 0;
  for (const res of results) { samples.push(...res.samples); plies += res.plies; if (res.terminal === "checkmate" || res.terminal === "adjudicated") decisive++; }
  return {
    samples, decisive, games, avgPlies: plies / games,
    batches: ev.batches, avgBatch: ev.batches ? ev.positions / ev.batches : 0,
  };
}

// ── Replay buffer ──────────────────────────────────────────────────────────
/** Sliding-window experience buffer — sample minibatches, drop oldest. */
export class ReplayBuffer {
  private buf: AZSample[] = [];
  constructor(private readonly capacity = 8000) {}
  get size(): number { return this.buf.length; }
  push(samples: AZSample[]): void {
    for (const s of samples) this.buf.push(s);
    if (this.buf.length > this.capacity) this.buf.splice(0, this.buf.length - this.capacity);
  }
  sample(n: number, rng: () => number): AZSample[] {
    const k = Math.min(n, this.buf.length);
    const out: AZSample[] = [];
    for (let i = 0; i < k; i++) out.push(this.buf[Math.floor(rng() * this.buf.length)]!);
    return out;
  }
}

/** One optimiser step on AZ targets: policy CE to π (per-sample), value MSE to z (batched conv). */
export async function trainStepAZ(
  policy: MLPPolicy, value: ConvValueNet, samples: AZSample[],
  opt: { zeroGrad: () => void; step: () => Promise<void> }, opts: { valueCoef?: number } = {},
): Promise<{ loss: number; policyLoss: number; valueLoss: number }> {
  const valueCoef = opts.valueCoef ?? 1.0;
  const B = samples.length;

  // Policy: CE per position (moves scored independently, variable count).
  const polTerms: Tensor[] = [];
  for (const smp of samples) {
    const out = await policy.forward(smp.moveFeats);
    polTerms.push(await (await rowVec(smp.pi).times(await out.probs.log())).neg()); // −Σ π log p
  }
  const policyLoss = await (await sumT(polTerms)).hadamard(scalar(1 / B));

  // Value: ONE batched conv forward over all sample planes → [B,1].
  const planes = new Float64Array(B * PLANE_DIM);
  for (let i = 0; i < B; i++) planes.set(samples[i]!.planes, i * PLANE_DIM);
  const vB = await value.forwardPlanes(planes, B);          // [B,1]
  const zT = new Tensor({ shape: [B, 1], type: "float64" }); zT.view.set(samples.map((s) => s.z));
  const dv = await vB.minus(zT);
  const sq = await dv.hadamard(dv);                         // [B,1]
  const onesRow = new Tensor({ shape: [1, B], type: "float64" }); onesRow.view.fill(1);
  const valueLoss = await (await onesRow.times(sq)).hadamard(scalar(1 / B)); // [1,1] mean

  const loss = await policyLoss.plus(await valueLoss.hadamard(scalar(valueCoef)));
  opt.zeroGrad();
  await loss.backward();
  await opt.step();
  return {
    loss: (loss.view as Float64Array)[0]!,
    policyLoss: (policyLoss.view as Float64Array)[0]!,
    valueLoss: (valueLoss.view as Float64Array)[0]!,
  };
}
