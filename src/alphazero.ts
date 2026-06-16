"use strict";

/**
 * AlphaZero-style training (stage 2): self-play where every move is chosen by
 * MCTS, and the network learns to imitate the SEARCH, not its own raw moves.
 *
 *   policy target π = MCTS visit distribution at the root (search-improved policy)
 *   value  target z = game outcome (mover's perspective), softened by material
 *                     lead on capped/drawn games so the signal isn't all zeros.
 *
 *   loss = mean( −Σₐ πₐ·log p(a|s) )   +   c · mean( (V(s) − z)² )
 *
 * The whole point of the user's MCTS request: search is a policy-improvement
 * operator; distilling it into the net is what makes the net itself stronger.
 */

import { Tensor } from "@euriklis/mathematics/tensor";
import { runMcts } from "@euriklis/mcts";
import { chessEnv, neuralEvaluator } from "./mcts-player";
import { type Color, startState, generateMoves, makeMove, inCheck } from "./rules";
import { featureMatrix } from "./features";
import { positionFeatures, whiteMaterial, ValueNet } from "./value";
import { MLPPolicy } from "./policy";

const scalar = (c: number): Tensor => { const t = new Tensor({ shape: [1, 1], type: "float64" }); t.view[0] = c; return t; };
const rowVec = (xs: number[]): Tensor => { const t = new Tensor({ shape: [1, xs.length], type: "float64" }); t.view.set(xs); return t; };
const sumT = async (ts: Tensor[]): Promise<Tensor> => { let a = ts[0]!; for (let i = 1; i < ts.length; i++) a = await a.plus(ts[i]!); return a; };

export interface AZSample { moveFeats: number[][]; posFeat: number[]; pi: number[]; z: number; }

export interface AZRolloutOpts {
  numSimulations?: number; cPuct?: number; maxPlies?: number;
  dirichletAlpha?: number; materialScale?: number;
}

/** One MCTS self-play game → (state, π, z) samples. */
export async function selfPlayMcts(policy: MLPPolicy, value: ValueNet, rng: () => number, opts: AZRolloutOpts = {}) {
  const sims = opts.numSimulations ?? 24, maxPlies = opts.maxPlies ?? 45;
  const cPuct = opts.cPuct ?? 1.5, dir = opts.dirichletAlpha ?? 0.3, matScale = opts.materialScale ?? 0.2;

  const s = startState();
  const bw: number[] = [whiteMaterial(s.board)];
  const recs: { moveFeats: number[][]; posFeat: number[]; pi: number[]; mover: Color }[] = [];
  let result: Color | 0 = 0, terminal = "cap";

  for (let ply = 0; ply < maxPlies; ply++) {
    const moves = generateMoves(s);
    if (moves.length === 0) { result = inCheck(s) ? ((-s.turn) as Color) : 0; terminal = inCheck(s) ? "checkmate" : "stalemate"; break; }
    if (s.halfmove >= 100) { result = 0; terminal = "draw"; break; }
    // temperature 1 ⇒ res.policy is the normalised visit distribution; res.action is sampled from it.
    const res = await runMcts(chessEnv, neuralEvaluator(policy, value), s, {
      numSimulations: sims, cPuct, temperature: 1, dirichletAlpha: dir, backup: "negamax", rng,
    });
    recs.push({ moveFeats: featureMatrix(s, moves), posFeat: positionFeatures(s), pi: res.policy, mover: s.turn });
    makeMove(s, res.action);
    bw.push(whiteMaterial(s.board));
  }

  const T = bw.length - 1;
  const samples: AZSample[] = recs.map((r) => ({
    moveFeats: r.moveFeats, posFeat: r.posFeat, pi: r.pi,
    z: terminal === "checkmate" ? (result === r.mover ? 1 : -1) : Math.tanh(r.mover * bw[T]! * matScale),
  }));
  return { samples, result, terminal, plies: recs.length };
}

/** One optimiser step on AlphaZero targets: policy CE to π, value MSE to z. */
export async function trainStepAZ(
  policy: MLPPolicy, value: ValueNet, samples: AZSample[],
  opt: { zeroGrad: () => void; step: () => Promise<void> }, opts: { valueCoef?: number } = {},
): Promise<{ loss: number; policyLoss: number; valueLoss: number }> {
  const valueCoef = opts.valueCoef ?? 1.0;
  const polTerms: Tensor[] = [], valTerms: Tensor[] = [];
  for (const smp of samples) {
    const out = await policy.forward(smp.moveFeats);                 // probs [L,1]
    const ce = await (await rowVec(smp.pi).times(await out.probs.log())).neg(); // −Σ π log p  [1,1]
    polTerms.push(ce);
    const v = await value.forward(smp.posFeat);                       // [1,1]
    const d = await v.minus(scalar(smp.z));
    valTerms.push(await d.hadamard(d));
  }
  const invN = scalar(1 / samples.length);
  const policyLoss = await (await sumT(polTerms)).hadamard(invN);
  const valueLoss = await (await sumT(valTerms)).hadamard(invN);
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
