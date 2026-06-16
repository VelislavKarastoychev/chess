import { test, expect } from "bun:test";
import { Adam } from "@euriklis/mathematics/tensor";
import { parseFEN, generateMoves } from "../src/rules";
import { featureMatrix } from "../src/features";
import { MLPPolicy } from "../src/policy";
import { trainStep, mulberry32, sampleFromProbs, type Transition } from "../src/selfplay";
import { computeReturns } from "../src/selfplay";

/**
 * Mechanics test: with a toy reward "+1 if the move is a capture", the REINFORCE
 * machinery must shift the policy's mass onto capture moves. This isolates the
 * gradient plumbing (forward → log π → advantage → backward → Adam) from chess
 * strength, and runs fast & deterministically with a seeded RNG.
 */
test("REINFORCE shifts policy toward a toy reward (captures)", async () => {
  const s = parseFEN("r3k2r/8/8/3pp3/3PP3/8/8/R3K2R w KQkq - 0 1");
  const moves = generateMoves(s);
  const feats = featureMatrix(s, moves);
  const isCapture = moves.map((m) => s.board[m.to] !== 0 || m.flag === "ep");
  expect(isCapture.some(Boolean)).toBe(true); // sanity: captures exist

  const policy = new MLPPolicy({ seed: 3 });
  const opt = new Adam(policy.parameters(), { lr: 0.05 });
  const rng = mulberry32(7);

  const captureMass = async () => {
    const out = await policy.forward(feats);
    const p = Array.from(out.probs.view as Float64Array);
    return p.reduce((a, pi, i) => a + (isCapture[i] ? pi : 0), 0);
  };

  const before = await captureMass();

  for (let it = 0; it < 80; it++) {
    const out = await policy.forward(feats);
    const probs = Array.from(out.probs.view as Float64Array);
    const batch: Transition[] = [];
    for (let b = 0; b < 24; b++) {
      const idx = sampleFromProbs(probs, rng);
      batch.push({ moveFeats: feats, posFeat: [], chosenIdx: idx, G: isCapture[idx] ? 1 : 0, mover: 1 });
    }
    await trainStep(policy, null, batch, opt, { advBaseline: "mean" });
  }

  const after = await captureMass();
  expect(after).toBeGreaterThan(before + 0.2); // learning happened
  expect(after).toBeGreaterThan(0.6);          // strongly prefers captures now
});

/** Returns are dense: a clean material gain yields a positive return. */
test("computeReturns densifies with material shaping", () => {
  // 3 plies: white wins a pawn at ply 2 (bw 0 → +1), no terminal result.
  const decisions: Transition[] = [
    { moveFeats: [], posFeat: [], chosenIdx: 0, G: 0, mover: 1 },
    { moveFeats: [], posFeat: [], chosenIdx: 0, G: 0, mover: -1 },
    { moveFeats: [], posFeat: [], chosenIdx: 0, G: 0, mover: 1 },
  ];
  const bw = [0, 0, 1, 1]; // white grabs a pawn between ply 1→2
  computeReturns(decisions, bw, 0, { gamma: 1, pawnScale: 0.1 });
  expect(decisions[0]!.G).toBeCloseTo(0.1, 6);  // white's move 0 leads to +1 pawn
  expect(decisions[1]!.G).toBeCloseTo(-0.1, 6); // black's view of the same swing
  expect(decisions[2]!.G).toBeCloseTo(0, 6);    // nothing after
});
