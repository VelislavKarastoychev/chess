/**
 * AlphaZero-style training: MCTS self-play → distil the search into the net.
 * Warm-starts from the existing checkpoint so MCTS begins from a decent net,
 * then refines policy (toward MCTS visit counts) and value (toward outcomes).
 *
 * Run:  bun run examples/train-az.ts
 *       CHESS_AZ_SIMS=32 CHESS_AZ_ITERS=20 bun run examples/train-az.ts
 */
import { Adam } from "@euriklis/mathematics/tensor";
import { loadCheckpoint, restore, snapshot, saveCheckpoint } from "../src/model-io";
import { selfPlayMcts, trainStepAZ, type AZSample } from "../src/alphazero";
import { evaluateVsRandom, mulberry32 } from "../src/selfplay";

const ITERS = Number(process.env.CHESS_AZ_ITERS ?? 15);
const GAMES = Number(process.env.CHESS_AZ_GAMES ?? 4);
const SIMS = Number(process.env.CHESS_AZ_SIMS ?? 20);
const rng = mulberry32(31);

const { policy, value } = restore(await loadCheckpoint("checkpoints/best.json"));
const opt = new Adam([...policy.parameters(), ...value.parameters()], { lr: 0.003 });
console.log(`AlphaZero training — warm-started, ${ITERS} iters × ${GAMES} MCTS games × ${SIMS} sims/move`);

const base = await evaluateVsRandom(policy, mulberry32(1), 30, 100);
console.log(`base (raw policy) vs random: ${(base.score * 100).toFixed(1)}%  material ${base.material.toFixed(2)}\n`);

let best = base.material;
for (let it = 1; it <= ITERS; it++) {
  const batch: AZSample[] = [];
  let mates = 0, plies = 0;
  for (let g = 0; g < GAMES; g++) {
    const { samples, terminal, plies: pl } = await selfPlayMcts(policy, value, rng, {
      numSimulations: SIMS, maxPlies: 45, cPuct: 1.5, dirichletAlpha: 0.3,
    });
    batch.push(...samples);
    plies += pl;
    if (terminal === "checkmate") mates++;
  }
  const { loss, policyLoss, valueLoss } = await trainStepAZ(policy, value, batch, opt, { valueCoef: 1.0 });
  const ev = await evaluateVsRandom(policy, mulberry32(1), 24, 100);
  console.log(`iter ${String(it).padStart(2)}  loss ${loss.toFixed(3)} (CE ${policyLoss.toFixed(3)}/V ${valueLoss.toFixed(3)})  decisive ${mates}/${GAMES}  avgPlies ${(plies / GAMES).toFixed(0)}  ►  vs random ${(ev.score * 100).toFixed(1)}% material ${ev.material.toFixed(2)}`);
  if (ev.material > best) { best = ev.material; await saveCheckpoint("checkpoints/best.json", snapshot(policy, value, { iter: it, mode: "alphazero", ...ev })); }
}

const final = await evaluateVsRandom(policy, mulberry32(7), 40, 120);
console.log(`\nfinal (raw policy) vs random: ${(final.score * 100).toFixed(1)}%  material ${final.material.toFixed(2)}`);
console.log(`Δ from base: ${((final.score - base.score) * 100).toFixed(1)} pts, ${(final.material - base.material).toFixed(2)} pawns`);
await saveCheckpoint("checkpoints/az.json", snapshot(policy, value, { iter: ITERS, mode: "alphazero", ...final }));
console.log("saved → checkpoints/best.json (peak), checkpoints/az.json (final)");
process.exit(0);
