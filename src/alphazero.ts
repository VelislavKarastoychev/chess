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
import { type Color, type State, type Move, startState, generateMoves, makeMove, inCheck, positionKey } from "./rules";
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
  adjudicate?: boolean;       // false ⇒ play to a true terminal (learn to convert)
  temperatureMoves?: number;  // plies of τ=1 sampling before switching to greedy (τ→0)
}

/** One MCTS self-play game through a shared (batched) evaluator → samples. */
export async function selfPlayGame(
  evaluator: Evaluator<State, Move>, rng: () => number, opts: AZRolloutOpts = {},
): Promise<{ samples: AZSample[]; result: Color | 0; terminal: string; plies: number }> {
  const sims = opts.numSimulations ?? 100, maxPlies = opts.maxPlies ?? 160;
  const cPuct = opts.cPuct ?? 1.5, dir = opts.dirichletAlpha ?? 0.3, matScale = opts.materialScale ?? 0.2;
  const adjLead = opts.adjudicateLead ?? 6, adjPersist = opts.adjudicatePersist ?? 6;
  const adjudicate = opts.adjudicate ?? true, tempMoves = opts.temperatureMoves ?? 24;

  const s = startState();
  const recs: { moveFeats: number[][]; planes: Float64Array; pi: number[]; mover: Color }[] = [];
  let result: Color | 0 = 0, terminal = "truncated";
  let leadPlies = 0, leadColor: Color | 0 = 0;
  const repCount = new Map<string, number>(); // threefold-repetition tracking

  for (let ply = 0; ply < maxPlies; ply++) {
    const moves = generateMoves(s);
    if (moves.length === 0) { result = inCheck(s) ? ((-s.turn) as Color) : 0; terminal = inCheck(s) ? "checkmate" : "stalemate"; break; }
    if (s.halfmove >= 100) { result = 0; terminal = "draw50"; break; }
    const reps = (repCount.get(positionKey(s)) ?? 0) + 1;
    repCount.set(positionKey(s), reps);
    if (reps >= 3) { result = 0; terminal = "repetition"; break; }

    // Temperature schedule: explore early (τ=1, sample ∝ visits), then play the
    // deciding/endgame phase greedily (τ→0, argmax visits) for sharper π and
    // stronger play — AlphaZero's first-~30-plies-then-exploit schedule.
    const temperature = ply < tempMoves ? 1 : 0;
    const res = await runMcts(chessEnv, evaluator, s, {
      numSimulations: sims, cPuct, temperature, dirichletAlpha: dir, backup: "negamax", rng,
    });
    // `reps - 1` = prior occurrences of this position (0 first time, 1 second).
    recs.push({ moveFeats: featureMatrix(s, moves), planes: encodePlanes(s, reps - 1), pi: res.policy, mover: s.turn });
    makeMove(s, res.action);

    // Adjudication: a sustained decisive material lead counts as a win, so the
    // value head sees real ±1 outcomes instead of endless truncated games. NOTE
    // this re-injects a material bias into z and stops the game before mate, so
    // a fraction of games (adjudicate=false) play to a true terminal to teach
    // conversion — watch the tactical suite for endgame regressions.
    if (adjudicate) {
      const wm = whiteMaterial(s.board);
      const cur: Color | 0 = Math.abs(wm) >= adjLead ? (Math.sign(wm) as Color) : 0;
      if (cur !== 0 && cur === leadColor) leadPlies++;
      else { leadColor = cur; leadPlies = cur !== 0 ? 1 : 0; }
      if (leadPlies >= adjPersist) { result = leadColor as Color; terminal = "adjudicated"; break; }
    }
  }

  const finalMat = whiteMaterial(s.board);
  const decisive = terminal === "checkmate" || terminal === "adjudicated";
  // Real draws (stalemate / 50-move / threefold repetition) are z = 0 — the
  // true outcome — NOT a material proxy, so the net learns that shuffling a
  // won position away is worth nothing and is pushed to convert. Only a
  // genuinely unresolved maxPlies truncation keeps the soft material fallback.
  const isDraw = terminal === "stalemate" || terminal === "draw50" || terminal === "repetition";
  const samples: AZSample[] = recs.map((r) => ({
    moveFeats: r.moveFeats, planes: r.planes, pi: r.pi,
    z: decisive ? (result === r.mover ? 1 : -1) : isDraw ? 0 : Math.tanh(r.mover * finalMat * matScale),
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
  opts: AZRolloutOpts & { games?: number; maxBatch?: number; terminalFraction?: number } = {},
): Promise<SelfPlayBatchResult> {
  const games = opts.games ?? 16;
  const termFrac = opts.terminalFraction ?? 0.25; // share of games played to a true terminal
  const ev = new BatchedEvaluator(policy, value, opts.maxBatch ?? 256);
  ev.active = games;
  const evaluate = ev.evaluator();

  // Per-game deterministic rng stream derived from the base rng.
  const streams = Array.from({ length: games }, (_, g) => {
    let a = (1000003 * (g + 1)) ^ Math.floor(rng() * 0x7fffffff);
    return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  });

  let done = 0;
  const results = await Promise.all(streams.map((r, g) =>
    selfPlayGame(evaluate, r, { ...opts, adjudicate: g >= Math.round(games * termFrac) })
      .finally(() => { done++; ev.active = games - done; }),
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

  // Policy: ONE matmul over every sample's moves concatenated (moves are scored
  // independently), then a segmented softmax + CE per position — the expensive
  // GEMM is batched; only the tiny per-segment softmax/CE stay per-sample.
  const concat: number[][] = [];
  const segOff: number[] = [], segLen: number[] = [];
  for (const smp of samples) { segOff.push(concat.length); segLen.push(smp.moveFeats.length); for (const row of smp.moveFeats) concat.push(row); }
  const scoresT = await policy.scoresTensor(concat); // [ΣL, 1], differentiable
  const polTerms: Tensor[] = [];
  for (let i = 0; i < B; i++) {
    const seg = await scoresT.sliceRows(segOff[i]!, segLen[i]!);     // [L,1]
    const logp = await (await seg.softmax("column")).log();          // [L,1]
    polTerms.push(await (await rowVec(samples[i]!.pi).times(logp)).neg()); // −Σ π log p
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
