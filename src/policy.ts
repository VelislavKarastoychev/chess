"use strict";

/**
 * Move-scoring policies over a *set* of candidate moves.
 *
 * Input is a feature matrix X ∈ [L, F] (L legal moves, each an F-dim vector
 * from features.ts). Output is a score per move → softmax = π(a | s), a
 * probability distribution over the legal moves. This is the "set-to-policy"
 * head the user described: weights pick one of the candidate vectors.
 *
 *   MLPPolicy        — scores each move INDEPENDENTLY (simple, strong baseline).
 *   AttentionPolicy  — moves attend to EACH OTHER (a non-causal transformer
 *                      block), so a move is scored RELATIVE to its alternatives.
 *                      This is where the transformer earns its place.
 *
 * Both expose `.parameters()` for `new Adam(policy.parameters())`, so the same
 * objects drop straight into the REINFORCE / actor-critic self-play loop next.
 *
 * Tensor ops are async (parallel-capable); every forward is awaited.
 */

import { Tensor } from "@euriklis/mathematics/tensor";
import { TransformerBlock } from "@euriklis/mathematics/tensor";
import { FEATURE_DIM } from "./features";

export interface PolicyOutput {
  scores: Tensor; // [L, 1] raw scores (logits)
  probs: Tensor;  // [L, 1] softmax over the L moves
}

const proj = (rows: number, cols: number, seed: number): Tensor =>
  Tensor.random({
    shape: [rows, cols], type: "float64", seed,
    from: -Math.sqrt(1 / rows), to: Math.sqrt(1 / rows), requiresGrad: true,
  });

const onesCol = (n: number): Tensor => {
  const t = new Tensor({ shape: [n, 1], type: "float64" });
  t.view.fill(1);
  return t;
};

/** Build a [L, F] input tensor from a feature matrix (no grad needed on input). */
const toTensor = (rows: number[][]): Tensor => {
  const L = rows.length, F = rows[0]?.length ?? FEATURE_DIM;
  const t = new Tensor({ shape: [L, F], type: "float64" });
  const flat = t.view;
  for (let i = 0; i < L; i++) for (let j = 0; j < F; j++) flat[i * F + j] = rows[i]![j]!;
  return t;
};

// ---------------------------------------------------------------------------
/** Two-layer MLP: each move scored independently. score = relu(X·W1+b1)·W2+b2. */
export class MLPPolicy {
  W1: Tensor; b1: Tensor; W2: Tensor; b2: Tensor;
  readonly hidden: number;

  constructor(opts: { hidden?: number; inDim?: number; seed?: number } = {}) {
    const F = opts.inDim ?? FEATURE_DIM;
    const H = (this.hidden = opts.hidden ?? 32);
    const s = opts.seed ?? 1;
    this.W1 = proj(F, H, s + 1);
    this.b1 = Tensor.zeros({ shape: [1, H], type: "float64", requiresGrad: true });
    this.W2 = proj(H, 1, s + 2);
    this.b2 = Tensor.zeros({ shape: [1, 1], type: "float64", requiresGrad: true });
  }

  parameters(): Tensor[] { return [this.W1, this.b1, this.W2, this.b2]; }

  async forward(features: number[][]): Promise<PolicyOutput> {
    const L = features.length;
    const X = toTensor(features);
    const ones = onesCol(L);
    const h = await (await (await X.times(this.W1)).plus(await ones.times(this.b1))).relu();
    const scores = await (await h.times(this.W2)).plus(await ones.times(this.b2));
    const probs = await scores.softmax("column");
    return { scores, probs };
  }
}

// ---------------------------------------------------------------------------
/** Project features → d, let moves attend to each other, then score → softmax. */
export class AttentionPolicy {
  Win: Tensor; bIn: Tensor; block: TransformerBlock; Wout: Tensor; bOut: Tensor;
  readonly dModel: number;

  constructor(opts: { dModel?: number; heads?: number; inDim?: number; seed?: number } = {}) {
    const F = opts.inDim ?? FEATURE_DIM;
    const d = (this.dModel = opts.dModel ?? 32);
    const heads = opts.heads ?? 4;
    const s = opts.seed ?? 1;
    this.Win = proj(F, d, s + 1);
    this.bIn = Tensor.zeros({ shape: [1, d], type: "float64", requiresGrad: true });
    // Non-causal: every move sees every other move (an unordered set).
    this.block = new TransformerBlock({ dModel: d, heads, causal: false, seed: s + 10 });
    this.Wout = proj(d, 1, s + 2);
    this.bOut = Tensor.zeros({ shape: [1, 1], type: "float64", requiresGrad: true });
  }

  parameters(): Tensor[] {
    return [this.Win, this.bIn, ...this.block.parameters(), this.Wout, this.bOut];
  }

  async forward(features: number[][]): Promise<PolicyOutput> {
    const L = features.length;
    const X = toTensor(features);
    const ones = onesCol(L);
    const embed = await (await X.times(this.Win)).plus(await ones.times(this.bIn)); // [L, d]
    const ctx = await this.block.forward(embed);                                    // [L, d]
    const scores = await (await ctx.times(this.Wout)).plus(await ones.times(this.bOut));
    const probs = await scores.softmax("column");
    return { scores, probs };
  }
}

export type Policy = MLPPolicy | AttentionPolicy;
