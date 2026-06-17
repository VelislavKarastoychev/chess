/**
 * Long, resumable AlphaZero training (overhauled pipeline).
 *
 *   bun run examples/train-az-long.ts
 *   CHESS_AZ_TARGET=5000 CHESS_AZ_GAMES=32 CHESS_AZ_SIMS=60 bun run examples/train-az-long.ts
 *
 * Each iteration:
 *   1. self-play GAMES games concurrently (one BatchedEvaluator → the conv
 *      value net amortises over a batch of leaves per forward);
 *   2. push samples into a sliding replay buffer;
 *   3. take TRAIN_STEPS minibatch optimiser steps (policy CE + value MSE);
 *   4. periodically: tactical suite + vs-random sanity + a GATING match against
 *      the champion — promote `best.json` only if the candidate clears a margin.
 *
 *   best.json       — current champion (promoted by gating, not material)
 *   latest-az.json  — rolling resume point (meta.iter)
 *
 * Defaults are tuned for THROUGHPUT on the CPU tensor stack: the conv value
 * forward is ~8 ms/batch, so per-iteration wall-clock is dominated by the
 * number of MCTS leaf evals (games × plies × sims). maxPlies=160 + 120 sims is
 * ~5 min/iter (untrained games drag to full length); the defaults below
 * (sims 60, maxPlies 60) are ~40-50 s/iter — bump them up once the net is
 * decent and games end sooner. See the profiling note in the repo history.
 */
import { Adam } from "@euriklis/mathematics/tensor";
import { loadCheckpoint, restore, snapshot, saveCheckpoint, type Checkpoint } from "../src/model-io";
import { selfPlayBatch, trainStepAZ, ReplayBuffer } from "../src/alphazero";
import { runParallelSelfPlay } from "../src/parallel-selfplay";
import { playMatch, type Net } from "../src/gating";
import { tacticalSuite } from "../src/tactical";
import { evaluateVsRandom, mulberry32 } from "../src/selfplay";
import { MLPPolicy } from "../src/policy";
import { ConvValueNet } from "../src/value";

const TARGET = Number(process.env.CHESS_AZ_TARGET ?? 5000);
const GAMES = Number(process.env.CHESS_AZ_GAMES ?? 32);    // more concurrent games → bigger eval batches
const SIMS = Number(process.env.CHESS_AZ_SIMS ?? 60);      // ramp up once the net is decent
const BATCH = Number(process.env.CHESS_AZ_BATCH ?? 256);
const TRAIN_STEPS = Number(process.env.CHESS_AZ_TRAINSTEPS ?? 8);
const REPLAY_CAP = Number(process.env.CHESS_AZ_REPLAY ?? 8000);
const MAXPLIES = Number(process.env.CHESS_AZ_MAXPLIES ?? 60); // untrained games drag; cap short early
const EVAL_EVERY = Number(process.env.CHESS_AZ_EVALEVERY ?? 10);
const SAVE_EVERY = Number(process.env.CHESS_AZ_SAVEEVERY ?? 20);
const GATE_GAMES = Number(process.env.CHESS_AZ_GATEGAMES ?? 20);
const GATE_MARGIN = Number(process.env.CHESS_AZ_GATEMARGIN ?? 0.55);
const TEMP_MOVES = Number(process.env.CHESS_AZ_TEMPMOVES ?? 24);   // plies of τ=1 before greedy
const TERM_FRAC = Number(process.env.CHESS_AZ_TERMFRAC ?? 0.25);   // share of games run to a true terminal
const WORKERS = Number(process.env.CHESS_AZ_WORKERS ?? 1);         // >1 → multi-process self-play (use N cores)
const RESUME = "checkpoints/latest-az.json", BEST = "checkpoints/best.json";

const cloneNet = (policy: MLPPolicy, value: ConvValueNet): Net => {
  const r = restore(snapshot(policy, value));
  return { policy: r.policy, value: r.value };
};

// ── Resume ───────────────────────────────────────────────────────────────
let policy: MLPPolicy, value: ConvValueNet, startIter = 0;
const resumeExists = await Bun.file(RESUME).exists();
const bestExists = await Bun.file(BEST).exists();
if (resumeExists || bestExists) {
  const ckpt = await loadCheckpoint(resumeExists ? RESUME : BEST);
  const r = restore(ckpt);
  policy = r.policy; value = r.value;
  startIter = Number((ckpt.meta as Record<string, unknown>)?.iter ?? 0);
  console.log(`resume ${resumeExists ? RESUME : BEST} @ iter ${startIter}${r.freshValue ? "  (value net started fresh — incompatible/old checkpoint)" : ""}`);
} else {
  policy = new MLPPolicy({ hidden: 32 });
  value = new ConvValueNet({ channels: 16, hidden: 64 });
  console.log("fresh nets (no checkpoint found)");
}

const opt = new Adam([...policy.parameters(), ...value.parameters()], { lr: 0.002 });
const rng = mulberry32(1234 + startIter);
const buffer = new ReplayBuffer(REPLAY_CAP);
let best: Net = cloneNet(policy, value); // champion held in memory for gating

const clock = () => new Date().toLocaleTimeString("en-GB"); // HH:MM:SS
console.log(
  `[${clock()}] AZ overhaul — target ${TARGET}, ${GAMES} games × ${SIMS} sims, maxPlies ${MAXPLIES}, ` +
  `termFrac ${TERM_FRAC}, tempMoves ${TEMP_MOVES}, replay ${REPLAY_CAP}, batch ${BATCH}, ` +
  `${WORKERS > 1 ? `${WORKERS} self-play workers` : "single process"}`,
);

let lastEvalT = performance.now(), lastEvalIt = startIter; // for s/iter timing

for (let it = startIter + 1; it <= TARGET; it++) {
  const spOpts = {
    games: GAMES, numSimulations: SIMS, maxPlies: MAXPLIES, cPuct: 1.5, dirichletAlpha: 0.3, maxBatch: BATCH,
    temperatureMoves: TEMP_MOVES, terminalFraction: TERM_FRAC,
  };
  const sp = WORKERS > 1
    ? await runParallelSelfPlay(policy, value, it, { ...spOpts, workers: WORKERS })
    : await selfPlayBatch(policy, value, rng, spOpts);
  buffer.push(sp.samples);

  let loss = 0, pl = 0, vl = 0;
  const steps = Math.min(TRAIN_STEPS, buffer.size > 0 ? TRAIN_STEPS : 0);
  for (let t = 0; t < steps; t++) {
    const mb = buffer.sample(BATCH, rng);
    const r = await trainStepAZ(policy, value, mb, opt, { valueCoef: 1.0 });
    loss = r.loss; pl = r.policyLoss; vl = r.valueLoss;
  }

  if (it % EVAL_EVERY === 0) {
    const tac = await tacticalSuite(policy, value, { numSimulations: SIMS });
    const vr = await evaluateVsRandom(policy, mulberry32(1), 20, 100);
    const gate = await playMatch(cloneNet(policy, value), best, { games: GATE_GAMES, numSimulations: SIMS, maxPlies: MAXPLIES, rng: mulberry32(7 + it) });
    const promoted = gate.aScore >= GATE_MARGIN;
    if (promoted) { best = cloneNet(policy, value); await saveCheckpoint(BEST, snapshot(policy, value, { iter: it, mode: "alphazero", gateScore: gate.aScore, tactical: tac.passed })); }
    const nowT = performance.now();
    const perIter = (nowT - lastEvalT) / 1000 / (it - lastEvalIt); // wall seconds / iteration over this window
    lastEvalT = nowT; lastEvalIt = it;
    console.log(
      `[${clock()} ${perIter.toFixed(1)}s/it] iter ${String(it).padStart(5)}/${TARGET}  loss ${loss.toFixed(3)} (CE ${pl.toFixed(3)}/V ${vl.toFixed(3)})  ` +
      `decisive ${sp.decisive}/${GAMES} avgPlies ${sp.avgPlies.toFixed(0)} batch≈${sp.avgBatch.toFixed(1)}  ` +
      `tactic ${tac.passed}/${tac.total}  vsRand ${(vr.score * 100).toFixed(0)}%  gate ${(gate.aScore * 100).toFixed(0)}%${promoted ? " *PROMOTED" : ""}`,
    );
  }
  if (it % SAVE_EVERY === 0) await saveCheckpoint(RESUME, snapshot(policy, value, { iter: it } as Checkpoint["meta"]));
}

await saveCheckpoint(RESUME, snapshot(policy, value, { iter: TARGET }));
console.log(`done — reached ${TARGET} iters.`);
process.exit(0);
