/**
 * Long, resumable AlphaZero training. Checkpoints frequently so a multi-hour
 * run survives interruption — relaunch and it continues from latest-az.json.
 *
 *   bun run examples/train-az-long.ts                       # target 10000 iters
 *   CHESS_AZ_TARGET=10000 CHESS_AZ_SIMS=20 bun run examples/train-az-long.ts
 *
 *   best.json       — best model so far (by material vs random)
 *   latest-az.json  — rolling resume point (iteration counter in meta.iter)
 */
import { Adam } from "@euriklis/mathematics/tensor";
import { loadCheckpoint, restore, snapshot, saveCheckpoint } from "../src/model-io";
import { selfPlayMcts, trainStepAZ, type AZSample } from "../src/alphazero";
import { evaluateVsRandom, mulberry32 } from "../src/selfplay";

const TARGET = Number(process.env.CHESS_AZ_TARGET ?? 10000);
const GAMES = Number(process.env.CHESS_AZ_GAMES ?? 4);
const SIMS = Number(process.env.CHESS_AZ_SIMS ?? 20);
const EVAL_EVERY = 25, SAVE_EVERY = 50;
const RESUME = "checkpoints/latest-az.json", BEST = "checkpoints/best.json";

const resumeExists = await Bun.file(RESUME).exists();
const ckpt = await loadCheckpoint(resumeExists ? RESUME : BEST);
const { policy, value } = restore(ckpt);
const startIter = Number((ckpt.meta as any)?.iter ?? 0);
let bestMaterial = Number((ckpt.meta as any)?.bestMaterial ?? -Infinity);
const opt = new Adam([...policy.parameters(), ...value.parameters()], { lr: 0.002 });
const rng = mulberry32(1234 + startIter);

console.log(`AZ long training — resume ${resumeExists ? RESUME : BEST} @ iter ${startIter}, target ${TARGET}, ${GAMES} games × ${SIMS} sims`);
if (startIter === 0) {
  const b = await evaluateVsRandom(policy, mulberry32(1), 30, 100);
  bestMaterial = b.material;
  console.log(`base vs random: ${(b.score * 100).toFixed(1)}%  material ${b.material.toFixed(2)}`);
}

for (let it = startIter + 1; it <= TARGET; it++) {
  const batch: AZSample[] = [];
  let mates = 0;
  for (let g = 0; g < GAMES; g++) {
    const { samples, terminal } = await selfPlayMcts(policy, value, rng, { numSimulations: SIMS, maxPlies: 45, cPuct: 1.5, dirichletAlpha: 0.3 });
    batch.push(...samples);
    if (terminal === "checkmate") mates++;
  }
  const { loss, policyLoss, valueLoss } = await trainStepAZ(policy, value, batch, opt, { valueCoef: 1.0 });

  if (it % EVAL_EVERY === 0) {
    const ev = await evaluateVsRandom(policy, mulberry32(1), 24, 100);
    const star = ev.material > bestMaterial ? " *best" : "";
    console.log(`iter ${String(it).padStart(5)}/${TARGET}  loss ${loss.toFixed(3)} (CE ${policyLoss.toFixed(3)}/V ${valueLoss.toFixed(3)})  decisive ${mates}/${GAMES}  vs random ${(ev.score * 100).toFixed(1)}% material ${ev.material.toFixed(2)}${star}`);
    if (ev.material > bestMaterial) { bestMaterial = ev.material; await saveCheckpoint(BEST, snapshot(policy, value, { iter: it, mode: "alphazero", bestMaterial, ...ev })); }
  }
  if (it % SAVE_EVERY === 0) await saveCheckpoint(RESUME, snapshot(policy, value, { iter: it, bestMaterial }));
}

await saveCheckpoint(RESUME, snapshot(policy, value, { iter: TARGET, bestMaterial }));
console.log(`done — reached ${TARGET} iters. best material ${bestMaterial.toFixed(2)} in ${BEST}`);
process.exit(0);
