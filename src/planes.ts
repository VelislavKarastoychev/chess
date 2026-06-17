"use strict";

/**
 * Board → tensor planes for the convolutional value net.
 *
 * AlphaZero-style spatial encoding: a stack of 8×8 binary planes, one per
 * (piece-type, side) combination. Unlike the 10 aggregate scalars the old
 * ValueNet saw, this preserves WHERE every piece is — so a conv tower can read
 * king safety, pawn structure, passed pawns, piece coordination and hanging
 * pieces straight off the board.
 *
 * Mover-relative (canonical) orientation: the board is vertically flipped for
 * Black so the side to move always advances "up the board", and the mover's
 * pieces always land in planes 0–5, the opponent's in 6–11. One network then
 * serves both colours — exactly the colour-symmetry trick the move featuriser
 * already uses.
 *
 *   plane 0..5  : mover's   P N B R Q K
 *   plane 6..11 : opponent's P N B R Q K
 *
 * Layout is [C, H, W] row-major (channel-major), matching Tensor.conv2d's
 * [Cin, H, W] / [N, Cin, H, W] convention.
 */

import { type State, WHITE } from "./rules";

export const PLANES = 12;
export const BOARD_HW = 8 * 8;
export const PLANE_DIM = PLANES * BOARD_HW; // 768

/**
 * Write one position's planes into `out` at `offset` (length PLANE_DIM). The
 * slice MUST be zero on entry (a fresh Float64Array is; reused buffers must be
 * cleared by the caller). Returns `out` for chaining.
 */
export function encodePlanes(s: State, out?: Float64Array, offset = 0): Float64Array {
  const o = out ?? new Float64Array(PLANE_DIM);
  const mover = s.turn;
  const b = s.board;
  for (let sq = 0; sq < 64; sq++) {
    const p = b[sq]!;
    if (p === 0) continue;
    const type = Math.abs(p);                       // 1..6
    const isMover = Math.sign(p) === mover ? 0 : 6;  // mover planes 0-5, foe 6-11
    const f = sq & 7;
    const r = sq >> 3;
    const rr = mover === WHITE ? r : 7 - r;          // flip vertically for Black
    o[offset + (isMover + type - 1) * BOARD_HW + rr * 8 + f] = 1;
  }
  return o;
}

/** Stack `states` into one [B·PLANE_DIM] buffer (= a [B,12,8,8] tensor). */
export function encodeBatch(states: State[]): Float64Array {
  const out = new Float64Array(states.length * PLANE_DIM);
  for (let i = 0; i < states.length; i++) encodePlanes(states[i]!, out, i * PLANE_DIM);
  return out;
}
