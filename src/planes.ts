"use strict";

/**
 * Board → tensor planes for the convolutional value net.
 *
 * AlphaZero-style spatial encoding: a stack of 8×8 planes. Twelve binary
 * piece planes preserve WHERE every piece is (so a conv tower can read king
 * safety, pawn structure, passed pawns and hanging pieces directly), plus six
 * "state" planes for facts the piece placement alone can't express — castling
 * rights, the en-passant target, and the halfmove clock. King safety hinges on
 * whether castling is still available; drawishness hinges on the 50-move
 * counter — so these are real signals, not decoration.
 *
 *   plane 0..5   : mover's    P N B R Q K
 *   plane 6..11  : opponent's P N B R Q K
 *   plane 12     : mover kingside castling right    (constant 0/1)
 *   plane 13     : mover queenside castling right   (constant 0/1)
 *   plane 14     : opponent kingside castling right (constant 0/1)
 *   plane 15     : opponent queenside castling right(constant 0/1)
 *   plane 16     : en-passant target square         (single 1, mover-relative)
 *   plane 17     : halfmove clock / 100             (constant, 50-move proximity)
 *
 * Mover-relative (canonical) orientation: the board is vertically flipped for
 * Black so the side to move always advances "up", and the mover's pieces /
 * rights always land in the lower planes. One network serves both colours.
 *
 * Layout is [C, H, W] row-major, matching Tensor.conv2d's [Cin,H,W] /
 * [N,Cin,H,W] convention.
 */

import { type State, WHITE, C_WK, C_WQ, C_BK, C_BQ } from "./rules";

export const PLANES = 18;
export const BOARD_HW = 8 * 8;
export const PLANE_DIM = PLANES * BOARD_HW; // 1152

/**
 * Write one position's planes into `out` at `offset` (length PLANE_DIM). The
 * slice MUST be zero on entry (a fresh Float64Array is; reused buffers must be
 * cleared by the caller). Returns `out` for chaining.
 */
export function encodePlanes(s: State, out?: Float64Array, offset = 0): Float64Array {
  const o = out ?? new Float64Array(PLANE_DIM);
  const mover = s.turn;
  const white = mover === WHITE;
  const b = s.board;
  // Mover-relative square (vertical flip for Black).
  const mrel = (sq: number): number => {
    const f = sq & 7, r = sq >> 3;
    return (white ? r : 7 - r) * 8 + f;
  };

  // 0..11 — piece planes
  for (let sq = 0; sq < 64; sq++) {
    const p = b[sq]!;
    if (p === 0) continue;
    const type = Math.abs(p);                       // 1..6
    const base = Math.sign(p) === mover ? 0 : 6;     // mover 0-5, foe 6-11
    o[offset + (base + type - 1) * BOARD_HW + mrel(sq)] = 1;
  }

  // 12..15 — castling rights, mover-relative, as constant planes
  const moverK = white ? C_WK : C_BK, moverQ = white ? C_WQ : C_BQ;
  const oppK = white ? C_BK : C_WK, oppQ = white ? C_BQ : C_WQ;
  const rights = [moverK, moverQ, oppK, oppQ];
  for (let i = 0; i < 4; i++) {
    if (s.castling & rights[i]!) {
      const planeBase = offset + (12 + i) * BOARD_HW;
      for (let k = 0; k < BOARD_HW; k++) o[planeBase + k] = 1;
    }
  }

  // 16 — en-passant target square (single 1)
  if (s.ep >= 0) o[offset + 16 * BOARD_HW + mrel(s.ep)] = 1;

  // 17 — halfmove clock proximity to the 50-move draw (constant plane)
  const hm = Math.min(s.halfmove / 100, 1);
  if (hm > 0) {
    const planeBase = offset + 17 * BOARD_HW;
    for (let k = 0; k < BOARD_HW; k++) o[planeBase + k] = hm;
  }

  return o;
}

/** Stack `states` into one [B·PLANE_DIM] buffer (= a [B,PLANES,8,8] tensor). */
export function encodeBatch(states: State[]): Float64Array {
  const out = new Float64Array(states.length * PLANE_DIM);
  for (let i = 0; i < states.length; i++) encodePlanes(states[i]!, out, i * PLANE_DIM);
  return out;
}
