/**
 * Step 4 end-to-end: train the move-scoring policy by self-play with an
 * actor-critic (REINFORCE + value baseline), measuring progress against a
 * random opponent. Material-shaped returns give a gradient every move, so the
 * policy learns to win material and stop hanging pieces without needing to
 * force mate first.
 *
 * Run:  bun run examples/train.ts
 */
import { Adam } from "@euriklis/mathematics/tensor";
import { MLPPolicy } from "../src/policy";
import { ValueNet } from "../src/value";
import { mulberry32, selfPlayGame, trainStep, evaluateVsRandom, type Transition } from "../src/selfplay";

const ITERS = 30, GAMES_PER_ITER = 6, MAX_PLIES = 50;
const rng = mulberry32(2026);

const policy = new MLPPolicy({ hidden: 32, seed: 5 });
const value = new ValueNet({ hidden: 24, seed: 99 });
const opt = new Adam([...policy.parameters(), ...value.parameters()], { lr: 0.01 });

const pCount = policy.parameters().reduce((a, p) => a + p.view.length, 0);
const vCount = value.parameters().reduce((a, p) => a + p.view.length, 0);
console.log(`MLPPolicy ${pCount} params + ValueNet ${vCount} params, self-play actor-critic`);

const before = await evaluateVsRandom(policy, mulberry32(1), 30, 80);
console.log(`\nbefore training — vs random: score ${(before.score * 100).toFixed(1)}%, avg material ${before.material.toFixed(2)}\n`);

for (let it = 1; it <= ITERS; it++) {
  const batch: Transition[] = [];
  let avgPlies = 0;
  for (let g = 0; g < GAMES_PER_ITER; g++) {
    const { decisions, plies } = await selfPlayGame(policy, rng, { maxPlies: MAX_PLIES, gamma: 0.98, pawnScale: 0.1, winScale: 1 });
    batch.push(...decisions);
    avgPlies += plies;
  }
  const { loss, policyLoss, valueLoss } = await trainStep(policy, value, batch, opt, { valueCoef: 0.5 });
  if (it % 5 === 0 || it === 1) {
    const meanG = batch.reduce((a, t) => a + t.G, 0) / batch.length;
    console.log(`iter ${String(it).padStart(2)}  loss ${loss.toFixed(4)} (π ${policyLoss.toFixed(3)}, V ${valueLoss.toFixed(3)})  meanReturn ${meanG.toFixed(3)}  avgPlies ${(avgPlies / GAMES_PER_ITER).toFixed(0)}  transitions ${batch.length}`);
  }
}

const after = await evaluateVsRandom(policy, mulberry32(1), 30, 80);
console.log(`\nafter training  — vs random: score ${(after.score * 100).toFixed(1)}%, avg material ${after.material.toFixed(2)}`);
console.log(`Δ score ${((after.score - before.score) * 100).toFixed(1)} pts,  Δ material ${(after.material - before.material).toFixed(2)} pawns`);
process.exit(0);
