import { test, expect } from "bun:test";
import { startState, makeMove, parseUci, positionKey } from "../src/rules";

/** A knight shuffle Nf3 Nc6 Ng1 Nb8 returns to the starting position. */
test("positionKey repeats after a knight shuffle back to start", () => {
  const s = startState();
  const k0 = positionKey(s);
  for (const uci of ["g1f3", "b8c6", "f3g1", "c6b8"]) {
    const m = parseUci(s, uci)!;
    expect(m).not.toBeNull();
    makeMove(s, m);
  }
  expect(positionKey(s)).toBe(k0); // same position (board + turn + castling + ep)
});

test("positionKey differs for different positions / side to move", () => {
  const a = startState();
  const b = startState();
  makeMove(b, parseUci(b, "e2e4")!);
  expect(positionKey(a)).not.toBe(positionKey(b)); // different board + ep + turn
});

/** Threefold detection: the same key seen 3 times = draw (the rule self-play uses). */
test("threefold repetition is detected by counting position keys", () => {
  const s = startState();
  const count = new Map<string, number>();
  const bump = (): number => {
    const r = (count.get(positionKey(s)) ?? 0) + 1;
    count.set(positionKey(s), r);
    return r;
  };
  expect(bump()).toBe(1); // start, 1st occurrence
  const shuffle = () => { for (const u of ["g1f3", "b8c6", "f3g1", "c6b8"]) makeMove(s, parseUci(s, u)!); };
  shuffle();
  expect(bump()).toBe(2); // back to start, 2nd
  shuffle();
  expect(bump()).toBe(3); // back to start, 3rd → draw by repetition
});

import { encodePlanes, PLANES, BOARD_HW } from "../src/planes";

test("repetition plane (18) encodes prior-occurrence count", () => {
  expect(PLANES).toBe(19);
  const s = startState();
  const planeSum = (buf: Float64Array, plane: number) => { let x = 0; for (let k = 0; k < BOARD_HW; k++) x += buf[plane * BOARD_HW + k]!; return x; };
  expect(planeSum(encodePlanes(s, 0), 18)).toBe(0);            // first occurrence → empty
  expect(planeSum(encodePlanes(s, 1), 18)).toBeCloseTo(0.5 * 64); // seen once → 0.5
  expect(planeSum(encodePlanes(s, 2), 18)).toBeCloseTo(1.0 * 64); // seen twice → 1.0
  expect(planeSum(encodePlanes(s, 5), 18)).toBeCloseTo(1.0 * 64); // capped at 2
  // piece planes unaffected by reps
  expect(planeSum(encodePlanes(s, 0), 0)).toBe(planeSum(encodePlanes(s, 2), 0));
});
