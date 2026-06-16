import { test, expect } from "bun:test";
import { parseFEN, startState, perft } from "../src/rules";

// Published perft node counts — the standard move-generation correctness oracle.
// https://www.chessprogramming.org/Perft_Results

test("perft startpos", () => {
  const s = startState();
  expect(perft(s, 1)).toBe(20);
  expect(perft(s, 2)).toBe(400);
  expect(perft(s, 3)).toBe(8902);
  expect(perft(s, 4)).toBe(197281);
});

test("perft Kiwipete (castling, ep, pins, promotions)", () => {
  const s = parseFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  expect(perft(s, 1)).toBe(48);
  expect(perft(s, 2)).toBe(2039);
  expect(perft(s, 3)).toBe(97862);
});

test("perft position 3 (en-passant edge cases)", () => {
  const s = parseFEN("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
  expect(perft(s, 1)).toBe(14);
  expect(perft(s, 2)).toBe(191);
  expect(perft(s, 3)).toBe(2812);
  expect(perft(s, 4)).toBe(43238);
});

test("perft position 5 (promotions galore)", () => {
  const s = parseFEN("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
  expect(perft(s, 1)).toBe(44);
  expect(perft(s, 2)).toBe(1486);
  expect(perft(s, 3)).toBe(62379);
});
