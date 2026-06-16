/**
 * A serious self-play training run: many iterations, periodic evaluation vs a
 * random opponent, an lr warm-down, and a saved checkpoint. Ends by letting the
 * trained policy play a full greedy game so you can watch it move.
 *
 * Run:  bun run examples/train-long.ts
 */
import { Adam } from "@euriklis/mathematics/tensor";
import { MLPPolicy } from "../src/policy";
import { ValueNet } from "../src/value";
import {
  mulberry32, selfPlayGame, trainStep, evaluateVsRandom, type Transition,
} from "../src/selfplay";
import { snapshot, saveCheckpoint } from "../src/model-io";
import { startState, generateMoves, makeMove, gameStatus, moveToUci, inCheck } from "../src/rules";

const ITERS = 200, GAMES_PER_ITER = 8, MAX_PLIES = 60;
const rng = mulberry32(2026);
const policy = new MLPPolicy({ hidden: 48, seed: 5 });
const value = new ValueNet({ hidden: 32, seed: 99 });
const opt = new Adam([...policy.parameters(), ...value.parameters()], { lr: 0.012 });

console.log(`self-play actor-critic — ${ITERS} iters × ${GAMES_PER_ITER} games, ` +
  `${policy.parameters().reduce((a, p) => a + p.view.length, 0)} policy params`);

const base = await evaluateVsRandom(policy, mulberry32(1), 40, 100);
console.log(`base  vs random: ${(base.score * 100).toFixed(1)}%  material ${base.material.toFixed(2)}\n`);

let best = -Infinity;
for (let it = 1; it <= ITERS; it++) {
  // lr warm-down over the run.
  (opt as unknown as { lr: number }).lr = 0.012 * (1 - 0.7 * (it / ITERS));

  const batch: Transition[] = [];
  for (let g = 0; g < GAMES_PER_ITER; g++) {
    const { decisions } = await selfPlayGame(policy, rng, { maxPlies: MAX_PLIES, gamma: 0.98, pawnScale: 0.1, winScale: 1.5 });
    batch.push(...decisions);
  }
  const { loss, policyLoss, valueLoss } = await trainStep(policy, value, batch, opt, { valueCoef: 0.5 });

  if (it % 10 === 0) {
    const meanG = batch.reduce((a, t) => a + t.G, 0) / batch.length;
    process.stdout.write(`iter ${String(it).padStart(3)}  loss ${loss.toFixed(3)} (π ${policyLoss.toFixed(3)}/V ${valueLoss.toFixed(3)})  meanReturn ${meanG.toFixed(3)}`);
  }
  if (it % 25 === 0) {
    const ev = await evaluateVsRandom(policy, mulberry32(1), 40, 100);
    console.log(`   ►  vs random ${(ev.score * 100).toFixed(1)}%  material ${ev.material.toFixed(2)}`);
    if (ev.material > best) { best = ev.material; await saveCheckpoint("checkpoints/best.json", snapshot(policy, value, { iter: it, ...ev })); }
  } else if (it % 10 === 0) console.log("");
}

const final = await evaluateVsRandom(policy, mulberry32(7), 60, 120);
console.log(`\nfinal vs random: ${(final.score * 100).toFixed(1)}%  material ${final.material.toFixed(2)}`);
console.log(`Δ from base: ${((final.score - base.score) * 100).toFixed(1)} pts score, ${(final.material - base.material).toFixed(2)} pawns material`);
await saveCheckpoint("checkpoints/final.json", snapshot(policy, value, { iter: ITERS, ...final }));
console.log("saved → checkpoints/final.json, checkpoints/best.json");

// Watch the trained policy play a greedy game (vs itself).
console.log("\ntrained policy, greedy self-play game:");
const s = startState();
const sans: string[] = [];
for (let ply = 0; ply < 80; ply++) {
  const moves = generateMoves(s);
  if (moves.length === 0) break;
  const out = await policy.forward((await import("../src/features")).featureMatrix(s, moves));
  const probs = Array.from(out.probs.view as Float64Array);
  let bi = 0; for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[bi]!) bi = i;
  sans.push(moveToUci(moves[bi]!));
  makeMove(s, moves[bi]!);
  if (s.halfmove >= 100) break;
}
console.log(sans.join(" "));
console.log(`end: ${gameStatus(s)}${inCheck(s) ? " (in check)" : ""}`);
process.exit(0);
