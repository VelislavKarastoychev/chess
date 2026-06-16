"use strict";

/**
 * Move featuriser — `f(state, move) → number[]` of fixed length FEATURE_DIM.
 *
 * This is the heart of the user's idea: instead of letting a language model
 * discover everything from raw move tokens, each *legal* candidate move is
 * described by a hand-designed vector of tactical/positional signals (does it
 * capture, give check, gain space, open a line, hang the piece, develop…).
 * The policy network then scores these vectors. Features do the heavy lifting,
 * so a small net can work — ideal for a CPU tensor library.
 *
 * All ranks are taken from the MOVER's perspective (white as-is, black
 * mirrored) so the feature is colour-symmetric and both sides share weights.
 *
 * Start small and grow: every entry is documented; append new signals at the
 * end and bump FEATURE_DIM. Nothing here mutates `state` (make+unmake only).
 */

import {
  type State, type Move, type Color,
  WHITE, P, N, B, R, Q, K,
  fileOf, rankOf, sq, pieceValue, kingSquare,
  isSquareAttacked, makeMove, unmakeMove,
} from "./rules";

/** Number of squares a piece of `type`/`color` pseudo-reaches from `from`. */
function reachCount(board: Int8Array, from: number, type: number, color: Color): number {
  const f = fileOf(from), r = rankOf(from);
  const on = (x: number, y: number) => x >= 0 && x < 8 && y >= 0 && y < 8;
  let n = 0;
  const step = (df: number, dr: number, slide: boolean) => {
    let cf = f + df, cr = r + dr;
    while (on(cf, cr)) {
      const t = board[sq(cf, cr)]!;
      if (t === 0) n++;
      else { if (Math.sign(t) === -color) n++; break; }
      if (!slide) break;
      cf += df; cr += dr;
    }
  };
  const DIAG = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const ORTH = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const KN = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  if (type === N) for (const [df, dr] of KN) step(df, dr, false);
  else if (type === K) for (const d of [...DIAG, ...ORTH]) step(d[0]!, d[1]!, false);
  else if (type === B) for (const d of DIAG) step(d[0]!, d[1]!, true);
  else if (type === R) for (const d of ORTH) step(d[0]!, d[1]!, true);
  else if (type === Q) for (const d of [...DIAG, ...ORTH]) step(d[0]!, d[1]!, true);
  return n;
}

/** Heuristic: does vacating `from` open a friendly slider's line ("opens a diagonal/file")? */
function opensLine(board: Int8Array, from: number, color: Color): boolean {
  const f = fileOf(from), r = rankOf(from);
  const on = (x: number, y: number) => x >= 0 && x < 8 && y >= 0 && y < 8;
  // For each axis, if one side has a friendly slider whose ray was blocked by
  // `from` and the opposite side has somewhere to see into, the line opens.
  const axes: Array<[number, number, number[]]> = [
    [1, 1, [B, Q]], [1, 0, [R, Q]], [0, 1, [R, Q]], [1, -1, [B, Q]],
  ];
  for (const [df, dr, sliders] of axes) {
    for (const sgn of [1, -1]) {
      let cf = f + df * sgn, cr = r + dr * sgn, blockerFound = false;
      while (on(cf, cr)) {
        const t = board[sq(cf, cr)]!;
        if (t !== 0) {
          if (Math.sign(t) === color && sliders.includes(Math.abs(t))) blockerFound = true;
          break;
        }
        cf += df * sgn; cr += dr * sgn;
      }
      // Opposite side must continue onto the board for the line to "open".
      if (blockerFound && on(f - df * sgn, r - dr * sgn)) return true;
    }
  }
  return false;
}

const CENTER = new Set([27, 28, 35, 36]);                 // d4 e4 d5 e5
const EXT_CENTER = new Set([                               // c3..f6 ring
  18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45,
]);

/** The 8 central squares whose control we track: c4 c5 d4 d5 e4 e5 f4 f5. */
export const CENTRAL = new Set([26, 34, 27, 35, 28, 36, 29, 37]);

/** How many of the CENTRAL squares a piece attacks/controls from `from`. */
export function centralControl(board: Int8Array, from: number, type: number, color: Color): number {
  const f = fileOf(from), r = rankOf(from);
  const on = (x: number, y: number) => x >= 0 && x < 8 && y >= 0 && y < 8;
  let n = 0;
  const mark = (s: number) => { if (CENTRAL.has(s)) n++; };
  if (type === P) {                                       // pawns control their two diagonals
    for (const df of [-1, 1]) if (on(f + df, r + color)) mark(sq(f + df, r + color));
    return n;
  }
  const DIAG = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const ORTH = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const KN = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  const step = (df: number, dr: number, slide: boolean) => {
    let cf = f + df, cr = r + dr;
    while (on(cf, cr)) {
      mark(sq(cf, cr));                                    // attacked (incl. the blocker square)
      if (board[sq(cf, cr)] !== 0 || !slide) break;
      cf += df; cr += dr;
    }
  };
  if (type === N) for (const [df, dr] of KN) step(df, dr, false);
  else if (type === K) for (const d of [...DIAG, ...ORTH]) step(d[0]!, d[1]!, false);
  else if (type === B) for (const d of DIAG) step(d[0]!, d[1]!, true);
  else if (type === R) for (const d of ORTH) step(d[0]!, d[1]!, true);
  else for (const d of [...DIAG, ...ORTH]) step(d[0]!, d[1]!, true); // queen
  return n;
}

/** Names of each feature slot, in order — keeps the vector self-documenting. */
export const FEATURE_NAMES = [
  "fromFile", "fromRankRel", "toFile", "toRankRel",
  "isP", "isN", "isB", "isR", "isQ", "isK",
  "movingValue", "isCapture", "capturedValue", "isPromotion",
  "givesCheck", "isCastle", "centerOccupy", "mobilityAfter",
  "landsAttacked", "landsDefended", "developsMinor", "pawnAdvance", "opensLine",
  "centerControlGain",
] as const;
export const FEATURE_DIM = FEATURE_NAMES.length;

/**
 * Featurise one legal move. `state.turn` is the mover. Pure: state is restored.
 */
export function moveFeatures(state: State, m: Move): number[] {
  const board = state.board;
  const mover = state.turn;
  const enemy = (-mover) as Color;
  const piece = board[m.from]!;
  const type = Math.abs(piece);

  // Mover-relative rank: white keeps rank, black mirrors (0<->7).
  const relRank = (s: number) => (mover === WHITE ? rankOf(s) : 7 - rankOf(s));

  const captured = m.flag === "ep" ? P : Math.abs(board[m.to]!);  // 0 if empty
  const isCapture = captured !== 0 ? 1 : 0;

  // Post-move signals (make → measure → unmake).
  const u = makeMove(state, m);
  const givesCheck = isSquareAttacked(board, kingSquare(board, enemy), mover) ? 1 : 0;
  const landsAttacked = isSquareAttacked(board, m.to, enemy) ? 1 : 0;
  const landsDefended = isSquareAttacked(board, m.to, mover) ? 1 : 0;
  const promoType = m.flag === "promo" ? m.promo : type;
  const mobilityAfter = reachCount(board, m.to, promoType, mover) / 27;
  const centerControlGain = centralControl(board, m.to, promoType, mover) / 8;
  unmakeMove(state, m, u);

  const isCastle = (m.flag === "castleK" || m.flag === "castleQ") ? 1 : 0;
  const backRank = mover === WHITE ? 0 : 7;
  const developsMinor = (type === N || type === B) && rankOf(m.from) === backRank ? 1 : 0;
  const pawnAdvance = type === P ? relRank(m.to) / 7 : 0;
  const center = CENTER.has(m.to) ? 1 : EXT_CENTER.has(m.to) ? 0.5 : 0;

  return [
    fileOf(m.from) / 7, relRank(m.from) / 7, fileOf(m.to) / 7, relRank(m.to) / 7,
    type === P ? 1 : 0, type === N ? 1 : 0, type === B ? 1 : 0,
    type === R ? 1 : 0, type === Q ? 1 : 0, type === K ? 1 : 0,
    pieceValue(piece) / 9, isCapture, pieceValue(captured) / 9, m.flag === "promo" ? 1 : 0,
    givesCheck, isCastle, center, mobilityAfter,
    landsAttacked, landsDefended, developsMinor, pawnAdvance, opensLine(board, m.from, mover) ? 1 : 0,
    centerControlGain,
  ];
}

/** Featurise every legal move of a position → matrix [L, FEATURE_DIM]. */
export function featureMatrix(state: State, moves: Move[]): number[][] {
  return moves.map((m) => moveFeatures(state, m));
}
