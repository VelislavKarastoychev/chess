"use strict";

/**
 * Critic V(s): a position → scalar value in (-1, 1), the baseline that tames
 * REINFORCE's variance (the "actor-critic" half). Tiny MLP over a handful of
 * position features — material is the dominant signal, so this learns fast.
 *
 * Features are taken from the side-to-move's perspective so one network serves
 * both colours, exactly like the move featuriser.
 */

import { Tensor } from "@euriklis/mathematics/tensor";
import {
  type State, type Color, P, N, B, R, Q,
  generateMoves, inCheck,
} from "./rules";
import { centralControl } from "./features";

/** Signed material balance from White's perspective (kings excluded). */
export function whiteMaterial(board: Int8Array): number {
  let bal = 0;
  const v = [0, 1, 3, 3, 5, 9, 0];
  for (let i = 0; i < 64; i++) bal += Math.sign(board[i]!) * v[Math.abs(board[i]!)]!;
  return bal;
}

export const POS_NAMES = [
  "matBalance", "pawnDiff", "knightDiff", "bishopDiff", "rookDiff", "queenDiff",
  "mobility", "inCheck", "phase", "centerControl",
] as const;
export const POS_DIM = POS_NAMES.length;

/** Net control of the 8 central squares, from `color`'s perspective. */
function centerControlBalance(board: Int8Array, color: Color): number {
  let net = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i]!;
    if (p === 0) continue;
    const c = Math.sign(p) as Color;
    net += (c === color ? 1 : -1) * centralControl(board, i, Math.abs(p), c);
  }
  return net;
}

/** Position features from the mover's perspective. */
export function positionFeatures(s: State): number[] {
  const b = s.board, c = s.turn;
  const cnt = (pt: number, col: Color) => {
    let n = 0;
    for (let i = 0; i < 64; i++) if (b[i] === col * pt) n++;
    return n;
  };
  const diff = (pt: number) => cnt(pt, c) - cnt(pt, (-c) as Color);
  const total = whiteMaterial(b);
  let mat = 0;
  for (let i = 0; i < 64; i++) mat += Math.abs([0, 1, 3, 3, 5, 9, 0][Math.abs(b[i]!)]!);
  return [
    (c * total) / 10,
    diff(P) / 8, diff(N) / 2, diff(B) / 2, diff(R) / 2, diff(Q) / 1,
    generateMoves(s).length / 40,
    inCheck(s) ? 1 : 0,
    mat / 78, // game phase (1 = full material)
    centerControlBalance(b, c) / 8,
  ];
}

const proj = (rows: number, cols: number, seed: number): Tensor =>
  Tensor.random({
    shape: [rows, cols], type: "float64", seed,
    from: -Math.sqrt(1 / rows), to: Math.sqrt(1 / rows), requiresGrad: true,
  });

/** v = tanh(relu(x·W1 + b1)·W2 + b2) ∈ (-1, 1). */
export class ValueNet {
  W1: Tensor; b1: Tensor; W2: Tensor; b2: Tensor;
  readonly hidden: number;

  constructor(opts: { hidden?: number; seed?: number } = {}) {
    const H = (this.hidden = opts.hidden ?? 24), s = opts.seed ?? 99;
    this.W1 = proj(POS_DIM, H, s + 1);
    this.b1 = Tensor.zeros({ shape: [1, H], type: "float64", requiresGrad: true });
    this.W2 = proj(H, 1, s + 2);
    this.b2 = Tensor.zeros({ shape: [1, 1], type: "float64", requiresGrad: true });
  }

  parameters(): Tensor[] { return [this.W1, this.b1, this.W2, this.b2]; }

  async forward(posFeat: number[]): Promise<Tensor> {
    const x = new Tensor({ shape: [1, POS_DIM], type: "float64" });
    x.view.set(posFeat);
    const h = await (await (await x.times(this.W1)).plus(this.b1)).relu();
    return await (await (await h.times(this.W2)).plus(this.b2)).tanh();
  }
}
