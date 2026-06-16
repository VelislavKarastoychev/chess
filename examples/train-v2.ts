/**
 * Stronger training (v2): opponent-league RL with reward shaping.
 *
 *  • Opponent pool = random + frozen snapshots of past policies (no mirror
 *    self-play collapse).
 *  • Rewards: material shaping + big checkmate bonus, and a PENALTY for
 *    stalemating/drawing a won position (the failure we saw vs the human).
 *  • Entropy regularisation keeps exploration; repetition is penalised.
 *
 * Run:  bun run examples/train-v2.ts
 */
import { Adam } from "@euriklis/mathematics/tensor";
import { MLPPolicy } from "../src/policy";
import { ValueNet } from "../src/value";
import {
  mulberry32, playVsOpponent, trainStep, evaluateVsRandom,
  randomOpponent, policyOpponent, clonePolicy, type Opponent, type Transition,
} from "../src/selfplay";
import { snapshot, saveCheckpoint } from "../src/model-io";

const ITERS = 100, GAMES = 8, MAX_PLIES = 80;
const rng = mulberry32(7);
const policy = new MLPPolicy({ hidden: 64, seed: 11 });
const value = new ValueNet({ hidden: 32, seed: 99 });
const opt = new Adam([...policy.parameters(), ...value.parameters()], { lr: 0.01 });
const frozen: MLPPolicy[] = [];   // league of past snapshots

console.log(`v2 league training — ${ITERS} iters × ${GAMES} games, ${policy.parameters().reduce((a, p) => a + p.view.length, 0)} policy params`);
const base = await evaluateVsRandom(policy, mulberry32(1), 40, 100);
console.log(`base vs random: ${(base.score * 100).toFixed(1)}%  material ${base.material.toFixed(2)}\n`);

let best = -Infinity;
for (let it = 1; it <= ITERS; it++) {
  (opt as unknown as { lr: number }).lr = 0.01 * (1 - 0.7 * (it / ITERS));
  const batch: Transition[] = [];
  let mates = 0, draws = 0;
  for (let g = 0; g < GAMES; g++) {
    const ourColor = g % 2 === 0 ? 1 : -1;
    // Half the games vs random, half vs a frozen snapshot (if any exist yet).
    const opp: Opponent = (rng() < 0.5 || frozen.length === 0)
      ? randomOpponent(rng)
      : policyOpponent(frozen[Math.floor(rng() * frozen.length)]!, rng, 0.4);
    const { transitions, terminal, result } = await playVsOpponent(policy, opp, ourColor, rng, {
      maxPlies: MAX_PLIES, gamma: 0.98, pawnScale: 0.1, winScale: 1, mateBonus: 1.5, drawPenalty: 0.3, repPenalty: 0.15,
    });
    batch.push(...transitions);
    if (terminal === "checkmate" && result === ourColor) mates++;
    if (terminal === "stalemate" || terminal === "draw50") draws++;
  }
  const { loss, policyLoss, valueLoss } = await trainStep(policy, value, batch, opt, { valueCoef: 0.5, entropyCoef: 0.01 });

  if (it % 10 === 0) {
    process.stdout.write(`iter ${String(it).padStart(3)}  loss ${loss.toFixed(3)} (π ${policyLoss.toFixed(3)}/V ${valueLoss.toFixed(3)})  wins ${mates}/${GAMES} stalemates ${draws}`);
  }
  if (it % 20 === 0) {
    frozen.push(clonePolicy(policy));
    if (frozen.length > 4) frozen.shift();
    const ev = await evaluateVsRandom(policy, mulberry32(1), 40, 120);
    console.log(`   ►  vs random ${(ev.score * 100).toFixed(1)}%  material ${ev.material.toFixed(2)}  (league ${frozen.length})`);
    if (ev.material > best) { best = ev.material; await saveCheckpoint("checkpoints/best.json", snapshot(policy, value, { iter: it, ...ev })); }
  } else if (it % 10 === 0) console.log("");
}

const final = await evaluateVsRandom(policy, mulberry32(7), 60, 150);
console.log(`\nfinal vs random: ${(final.score * 100).toFixed(1)}%  material ${final.material.toFixed(2)}`);
console.log(`Δ from base: ${((final.score - base.score) * 100).toFixed(1)} pts, ${(final.material - base.material).toFixed(2)} pawns`);
await saveCheckpoint("checkpoints/v2.json", snapshot(policy, value, { iter: ITERS, ...final }));
console.log("saved → checkpoints/best.json (peak), checkpoints/v2.json (final)");
process.exit(0);
