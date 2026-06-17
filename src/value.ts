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
import { PLANES, BOARD_HW, encodePlanes, encodeBatch } from "./planes";

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

// ───────────────────────────────────────────────────────────────────────────
// Convolutional value net — sees the raw board (12×8×8 planes) instead of 10
// aggregate scalars. This is the real ceiling lift: a conv tower can read king
// safety, pawn structure, passed pawns and hanging pieces directly off the
// squares. Batched forward ([B,12,8,8] → [B,1]) so MCTS leaf evaluation and
// training can amortise the (heavier) conv over many positions in one pass.
// ───────────────────────────────────────────────────────────────────────────

const convW = (cout: number, cin: number, k: number, seed: number): Tensor => {
  const fan = cin * k * k;
  return Tensor.random({ shape: [cout, cin, k, k], type: "float64", seed, from: -Math.sqrt(1 / fan), to: Math.sqrt(1 / fan), requiresGrad: true });
};

export interface ConvValueOpts { channels?: number; hidden?: number; seed?: number; }

export class ConvValueNet {
  readonly channels: number;
  readonly hidden: number;
  c1w: Tensor; c1b: Tensor;   // conv 12→C, 3×3 pad1
  c2w: Tensor; c2b: Tensor;   // conv C→C, 3×3 pad1
  fc1w: Tensor; fc1b: Tensor; // (C·64) → H
  fc2w: Tensor; fc2b: Tensor; // H → 1
  readonly kind = "conv" as const;

  constructor(opts: ConvValueOpts = {}) {
    const C = (this.channels = opts.channels ?? 16);
    const H = (this.hidden = opts.hidden ?? 64);
    const s = opts.seed ?? 99;
    this.c1w = convW(C, PLANES, 3, s + 1);
    this.c1b = Tensor.zeros({ shape: [C], type: "float64", requiresGrad: true });
    this.c2w = convW(C, C, 3, s + 2);
    this.c2b = Tensor.zeros({ shape: [C], type: "float64", requiresGrad: true });
    this.fc1w = proj(C * BOARD_HW, H, s + 3);
    this.fc1b = Tensor.zeros({ shape: [1, H], type: "float64", requiresGrad: true });
    this.fc2w = proj(H, 1, s + 4);
    this.fc2b = Tensor.zeros({ shape: [1, 1], type: "float64", requiresGrad: true });
  }

  parameters(): Tensor[] {
    return [this.c1w, this.c1b, this.c2w, this.c2b, this.fc1w, this.fc1b, this.fc2w, this.fc2b];
  }

  /** Batched value over a [B·PLANE_DIM] planes buffer → [B,1] in (-1,1). */
  async forwardPlanes(planes: Float64Array, B: number): Promise<Tensor> {
    const x = new Tensor({ shape: [B, PLANES, 8, 8], type: "float64" });
    x.view.set(planes);
    let h = await (await x.conv2d(this.c1w, this.c1b, { stride: 1, padding: 1 })).relu(); // [B,C,8,8]
    h = await (await h.conv2d(this.c2w, this.c2b, { stride: 1, padding: 1 })).relu();      // [B,C,8,8]
    const flat = await h.reshape([B, this.channels * BOARD_HW]);                            // [B, C·64]
    const ones = new Tensor({ shape: [B, 1], type: "float64" }); ones.view.fill(1);
    let f = await (await flat.times(this.fc1w)).plus(await ones.times(this.fc1b));          // [B,H]
    f = await f.relu();
    const v = await (await f.times(this.fc2w)).plus(await ones.times(this.fc2b));           // [B,1]
    return await v.tanh();
  }

  /** Value of one position (mover's perspective), [1,1]. */
  async forward(s: State): Promise<Tensor> {
    return await this.forwardPlanes(encodePlanes(s), 1);
  }

  /** Values of many positions in one batched forward → [B,1]. */
  async forwardStates(states: State[]): Promise<Tensor> {
    return await this.forwardPlanes(encodeBatch(states), states.length);
  }
}
