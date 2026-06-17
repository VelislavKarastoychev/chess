"use strict";

/**
 * Minimal but *correct* chess rules engine — the foundation both the policy
 * idea and any LM idea depend on. Mailbox 8×8 board, full legal move
 * generation (incl. castling, en-passant, promotion, pins via make+king-safety
 * filtering), make/unmake, and `perft` for validation.
 *
 * Board layout: a single Int8Array(64), index = rank*8 + file.
 *   a1 = 0,  h1 = 7,  a8 = 56,  h8 = 63.   file = idx & 7,  rank = idx >> 3.
 * Pieces are signed: white > 0, black < 0.  1=P 2=N 3=B 4=R 5=Q 6=K, 0=empty.
 * `turn` is the side to move: +1 = white, -1 = black (same sign as its pieces).
 *
 * Correctness is verified by `perft` against the published node counts for the
 * start position and "Kiwipete" (see tests/perft.test.ts).
 */

export const WHITE = 1;
export const BLACK = -1;
export type Color = 1 | -1;

export const P = 1, N = 2, B = 3, R = 4, Q = 5, K = 6;

/** Centipawn-ish piece values, indexed by |piece| (0 unused). */
export const PIECE_VALUE = [0, 1, 3, 3, 5, 9, 0] as const;
export const pieceValue = (p: number): number => PIECE_VALUE[Math.abs(p)]!;

// Castling-rights bitmask.
export const C_WK = 1, C_WQ = 2, C_BK = 4, C_BQ = 8;

export interface State {
  board: Int8Array; // 64
  turn: Color;
  castling: number; // C_* bitmask
  ep: number;       // en-passant TARGET square, or -1
  halfmove: number; // halfmove clock (50-move rule)
  fullmove: number;
}

export type MoveFlag =
  | "normal" | "double" | "ep" | "castleK" | "castleQ" | "promo";

export interface Move {
  from: number;
  to: number;
  promo: number;   // promotion piece type (N..Q) or 0
  flag: MoveFlag;
}

/** Undo record returned by makeMove and consumed by unmakeMove. */
interface Undo {
  captured: number;     // signed piece removed (0 if none), for ep this is the pawn
  capturedSq: number;   // square the captured piece sat on (differs from `to` for ep)
  castling: number;
  ep: number;
  halfmove: number;
}

// ---- square helpers ---------------------------------------------------------
export const fileOf = (s: number): number => s & 7;
export const rankOf = (s: number): number => s >> 3;
export const sq = (file: number, rank: number): number => rank * 8 + file;
const onBoard = (file: number, rank: number): boolean =>
  file >= 0 && file < 8 && rank >= 0 && rank < 8;

const KNIGHT_D: ReadonlyArray<[number, number]> = [
  [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2],
];
const KING_D: ReadonlyArray<[number, number]> = [
  [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1],
];
const BISHOP_D: ReadonlyArray<[number, number]> = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const ROOK_D: ReadonlyArray<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/** Square a piece of `by` color would lose a castling right on, AND-ed out. */
const castleMaskFor = (square: number): number => {
  switch (square) {
    case 0: return ~C_WQ;   // a1
    case 7: return ~C_WK;   // h1
    case 56: return ~C_BQ;  // a8
    case 63: return ~C_BK;  // h8
    case 4: return ~(C_WK | C_WQ);  // e1
    case 60: return ~(C_BK | C_BQ); // e8
    default: return ~0;
  }
};

// ---- attack detection -------------------------------------------------------
/** Is `square` attacked by any piece of color `by`? */
export function isSquareAttacked(board: Int8Array, square: number, by: Color): boolean {
  const f = fileOf(square), r = rankOf(square);

  // Pawns: a `by`-pawn attacking `square` sits one rank toward its own side.
  const pr = r - by;
  for (const df of [-1, 1]) {
    const pf = f + df;
    if (onBoard(pf, pr) && board[sq(pf, pr)] === by * P) return true;
  }
  // Knights & king (adjacency).
  for (const [df, dr] of KNIGHT_D) {
    if (onBoard(f + df, r + dr) && board[sq(f + df, r + dr)] === by * N) return true;
  }
  for (const [df, dr] of KING_D) {
    if (onBoard(f + df, r + dr) && board[sq(f + df, r + dr)] === by * K) return true;
  }
  // Sliding: bishop/queen on diagonals, rook/queen on files/ranks.
  for (const [df, dr] of BISHOP_D) {
    let cf = f + df, cr = r + dr;
    while (onBoard(cf, cr)) {
      const p = board[sq(cf, cr)]!;
      if (p !== 0) { if (p === by * B || p === by * Q) return true; break; }
      cf += df; cr += dr;
    }
  }
  for (const [df, dr] of ROOK_D) {
    let cf = f + df, cr = r + dr;
    while (onBoard(cf, cr)) {
      const p = board[sq(cf, cr)]!;
      if (p !== 0) { if (p === by * R || p === by * Q) return true; break; }
      cf += df; cr += dr;
    }
  }
  return false;
}

export function kingSquare(board: Int8Array, color: Color): number {
  const king = color * K;
  for (let i = 0; i < 64; i++) if (board[i] === king) return i;
  return -1;
}

export const inCheck = (s: State, color: Color = s.turn): boolean =>
  isSquareAttacked(s.board, kingSquare(s.board, color), (-color) as Color);

// ---- pseudo-legal generation ------------------------------------------------
function pseudoMoves(s: State): Move[] {
  const { board, turn } = s;
  const moves: Move[] = [];
  const add = (from: number, to: number, flag: MoveFlag = "normal", promo = 0) =>
    moves.push({ from, to, promo, flag });

  for (let from = 0; from < 64; from++) {
    const piece = board[from]!;
    if (piece === 0 || Math.sign(piece) !== turn) continue;
    const f = fileOf(from), r = rankOf(from), type = Math.abs(piece);

    if (type === P) {
      const dir = turn;                       // white +1 rank, black -1 rank
      const startRank = turn === WHITE ? 1 : 6;
      const promoRank = turn === WHITE ? 7 : 0;
      const one = sq(f, r + dir);
      if (onBoard(f, r + dir) && board[one] === 0) {
        if (r + dir === promoRank) for (const pr of [Q, R, B, N]) add(from, one, "promo", pr);
        else {
          add(from, one);
          const two = sq(f, r + 2 * dir);
          if (r === startRank && board[two] === 0) add(from, two, "double");
        }
      }
      for (const df of [-1, 1]) {
        const cf = f + df, cr = r + dir;
        if (!onBoard(cf, cr)) continue;
        const to = sq(cf, cr), target = board[to]!;
        if (target !== 0 && Math.sign(target) === -turn) {
          if (cr === promoRank) for (const pr of [Q, R, B, N]) add(from, to, "promo", pr);
          else add(from, to);
        } else if (to === s.ep) {
          add(from, to, "ep");
        }
      }
    } else if (type === N) {
      for (const [df, dr] of KNIGHT_D) {
        if (!onBoard(f + df, r + dr)) continue;
        const to = sq(f + df, r + dr), t = board[to]!;
        if (t === 0 || Math.sign(t) === -turn) add(from, to);
      }
    } else if (type === K) {
      for (const [df, dr] of KING_D) {
        if (!onBoard(f + df, r + dr)) continue;
        const to = sq(f + df, r + dr), t = board[to]!;
        if (t === 0 || Math.sign(t) === -turn) add(from, to);
      }
      // Castling: rights present, squares empty, king not in/through check.
      const enemy = (-turn) as Color;
      if (turn === WHITE && from === 4 && !isSquareAttacked(board, 4, enemy)) {
        if ((s.castling & C_WK) && board[5] === 0 && board[6] === 0 &&
            !isSquareAttacked(board, 5, enemy) && !isSquareAttacked(board, 6, enemy)) add(4, 6, "castleK");
        if ((s.castling & C_WQ) && board[3] === 0 && board[2] === 0 && board[1] === 0 &&
            !isSquareAttacked(board, 3, enemy) && !isSquareAttacked(board, 2, enemy)) add(4, 2, "castleQ");
      } else if (turn === BLACK && from === 60 && !isSquareAttacked(board, 60, enemy)) {
        if ((s.castling & C_BK) && board[61] === 0 && board[62] === 0 &&
            !isSquareAttacked(board, 61, enemy) && !isSquareAttacked(board, 62, enemy)) add(60, 62, "castleK");
        if ((s.castling & C_BQ) && board[59] === 0 && board[58] === 0 && board[57] === 0 &&
            !isSquareAttacked(board, 59, enemy) && !isSquareAttacked(board, 58, enemy)) add(60, 58, "castleQ");
      }
    } else {
      // Sliding pieces.
      const dirs = type === B ? BISHOP_D : type === R ? ROOK_D : [...BISHOP_D, ...ROOK_D];
      for (const [df, dr] of dirs) {
        let cf = f + df, cr = r + dr;
        while (onBoard(cf, cr)) {
          const to = sq(cf, cr), t = board[to]!;
          if (t === 0) add(from, to);
          else { if (Math.sign(t) === -turn) add(from, to); break; }
          cf += df; cr += dr;
        }
      }
    }
  }
  return moves;
}

// ---- make / unmake ----------------------------------------------------------
export function makeMove(s: State, m: Move): Undo {
  const { board } = s;
  const piece = board[m.from]!;
  const undo: Undo = {
    captured: 0, capturedSq: -1, castling: s.castling, ep: s.ep, halfmove: s.halfmove,
  };

  // Identify capture (en-passant captures off-square).
  if (m.flag === "ep") {
    const capSq = m.to - 8 * s.turn;
    undo.captured = board[capSq]!;
    undo.capturedSq = capSq;
    board[capSq] = 0;
  } else if (board[m.to] !== 0) {
    undo.captured = board[m.to]!;
    undo.capturedSq = m.to;
  }

  // Move the piece (handle promotion).
  board[m.to] = m.flag === "promo" ? (s.turn * m.promo) as number : piece;
  board[m.from] = 0;

  // Castling: move the rook too.
  if (m.flag === "castleK") {
    const rfrom = s.turn === WHITE ? 7 : 63, rto = s.turn === WHITE ? 5 : 61;
    board[rto] = board[rfrom]!; board[rfrom] = 0;
  } else if (m.flag === "castleQ") {
    const rfrom = s.turn === WHITE ? 0 : 56, rto = s.turn === WHITE ? 3 : 59;
    board[rto] = board[rfrom]!; board[rfrom] = 0;
  }

  // En-passant target: only set after a double push.
  s.ep = m.flag === "double" ? (m.from + m.to) / 2 : -1;

  // Castling rights: lose them if king/rook left home or a rook was captured.
  s.castling &= castleMaskFor(m.from) & castleMaskFor(m.to);

  // Clocks.
  s.halfmove = (Math.abs(piece) === P || undo.captured !== 0) ? 0 : s.halfmove + 1;
  if (s.turn === BLACK) s.fullmove++;
  s.turn = (-s.turn) as Color;
  return undo;
}

export function unmakeMove(s: State, m: Move, u: Undo): void {
  const { board } = s;
  s.turn = (-s.turn) as Color;
  if (s.turn === BLACK) s.fullmove--;
  s.castling = u.castling;
  s.ep = u.ep;
  s.halfmove = u.halfmove;

  const moved = m.flag === "promo" ? (s.turn * P) as number : board[m.to]!;
  board[m.from] = moved;
  board[m.to] = 0;

  if (u.captured !== 0) board[u.capturedSq] = u.captured;

  if (m.flag === "castleK") {
    const rfrom = s.turn === WHITE ? 7 : 63, rto = s.turn === WHITE ? 5 : 61;
    board[rfrom] = board[rto]!; board[rto] = 0;
  } else if (m.flag === "castleQ") {
    const rfrom = s.turn === WHITE ? 0 : 56, rto = s.turn === WHITE ? 3 : 59;
    board[rfrom] = board[rto]!; board[rto] = 0;
  }
}

// ---- legal generation (filter out king-in-check) ---------------------------
export function generateMoves(s: State): Move[] {
  const legal: Move[] = [];
  const mover = s.turn;
  for (const m of pseudoMoves(s)) {
    const u = makeMove(s, m);
    if (!isSquareAttacked(s.board, kingSquare(s.board, mover), s.turn)) legal.push(m);
    unmakeMove(s, m, u);
  }
  return legal;
}

export type Status = "ongoing" | "checkmate" | "stalemate" | "draw50";
export function gameStatus(s: State): Status {
  if (generateMoves(s).length === 0) return inCheck(s) ? "checkmate" : "stalemate";
  if (s.halfmove >= 100) return "draw50";
  return "ongoing";
}

/**
 * Compact key identifying a position for threefold-repetition detection: the
 * FIDE "same position" fields — piece placement, side to move, castling rights
 * and the en-passant target. Two states with the same key are the same
 * position; a key seen three times in a game is a draw by repetition.
 */
export function positionKey(s: State): string {
  let k = "";
  for (let i = 0; i < 64; i++) k += String.fromCharCode(s.board[i]! + 7); // -6..6 → 1..13
  return `${k}${s.turn === WHITE ? "w" : "b"},${s.castling},${s.ep}`;
}

// ---- FEN & display ----------------------------------------------------------
export const START_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const FEN_PIECE: Record<string, number> = {
  P, N, B, R, Q, K,
  p: -P, n: -N, b: -B, r: -R, q: -Q, k: -K,
};

export function parseFEN(fen: string): State {
  const [placement, turn, castling, ep, half, full] = fen.trim().split(/\s+/);
  const board = new Int8Array(64);
  let rank = 7, file = 0;
  for (const ch of placement!) {
    if (ch === "/") { rank--; file = 0; }
    else if (ch >= "1" && ch <= "8") file += +ch;
    else { board[sq(file, rank)] = FEN_PIECE[ch]!; file++; }
  }
  let cast = 0;
  if (castling!.includes("K")) cast |= C_WK;
  if (castling!.includes("Q")) cast |= C_WQ;
  if (castling!.includes("k")) cast |= C_BK;
  if (castling!.includes("q")) cast |= C_BQ;
  const epSq = ep && ep !== "-"
    ? sq(ep.charCodeAt(0) - 97, +ep[1]! - 1) : -1;
  return {
    board,
    turn: turn === "w" ? WHITE : BLACK,
    castling: cast,
    ep: epSq,
    halfmove: half ? +half : 0,
    fullmove: full ? +full : 1,
  };
}

export const startState = (): State => parseFEN(START_FEN);

const FILES = "abcdefgh";
export const squareName = (s: number): string => FILES[fileOf(s)]! + (rankOf(s) + 1);
export function moveToUci(m: Move): string {
  const promo = m.promo ? "nbrq"[m.promo - 2] : "";
  return squareName(m.from) + squareName(m.to) + promo;
}

// ---- board rendering & move parsing (for interactive play) ------------------
const GLYPH = [".", "P", "N", "B", "R", "Q", "K"];
/** ASCII board, rank 8 at top, White uppercase / Black lowercase. */
export function renderBoard(s: State): string {
  const rows: string[] = [];
  for (let r = 7; r >= 0; r--) {
    let row = `${r + 1} `;
    for (let f = 0; f < 8; f++) {
      const p = s.board[sq(f, r)]!;
      row += (p < 0 ? GLYPH[-p]!.toLowerCase() : GLYPH[p]!) + " ";
    }
    rows.push(row);
  }
  rows.push("  a b c d e f g h");
  return rows.join("\n");
}

/** Match a UCI string (e.g. "e2e4", "e7e8q") to a legal move, or null. */
export function parseUci(s: State, uci: string): Move | null {
  const u = uci.trim().toLowerCase();
  return generateMoves(s).find((m) => moveToUci(m) === u) ?? null;
}

/** Serialise a position back to FEN (for persisting a game between moves). */
export function toFEN(s: State): string {
  let placement = "";
  for (let r = 7; r >= 0; r--) {
    let empty = 0;
    for (let f = 0; f < 8; f++) {
      const p = s.board[sq(f, r)]!;
      if (p === 0) { empty++; continue; }
      if (empty) { placement += empty; empty = 0; }
      placement += p < 0 ? GLYPH[-p]!.toLowerCase() : GLYPH[p]!;
    }
    if (empty) placement += empty;
    if (r > 0) placement += "/";
  }
  let cast = "";
  if (s.castling & C_WK) cast += "K";
  if (s.castling & C_WQ) cast += "Q";
  if (s.castling & C_BK) cast += "k";
  if (s.castling & C_BQ) cast += "q";
  return `${placement} ${s.turn === WHITE ? "w" : "b"} ${cast || "-"} ` +
    `${s.ep >= 0 ? squareName(s.ep) : "-"} ${s.halfmove} ${s.fullmove}`;
}

// ---- perft (move-generation correctness oracle) -----------------------------
export function perft(s: State, depth: number): number {
  if (depth === 0) return 1;
  const moves = generateMoves(s);
  if (depth === 1) return moves.length;
  let nodes = 0;
  for (const m of moves) {
    const u = makeMove(s, m);
    nodes += perft(s, depth - 1);
    unmakeMove(s, m, u);
  }
  return nodes;
}
