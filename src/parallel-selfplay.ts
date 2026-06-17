"use strict";

/**
 * Multi-process self-play: fan game generation out across `workers` child
 * processes so a multi-core machine is actually used (the MCTS loop itself is
 * single-threaded JS, so one process pegs one core). Synchronous actor-learner:
 * each call snapshots the CURRENT net to a temp checkpoint, spawns the workers
 * (each plays games/workers games with that net), waits for all, and merges
 * their samples. The trainer then updates the net and the next call repeats —
 * so every worker always plays the iteration-start net (no stale-weight drift).
 *
 * Drop-in replacement for `selfPlayBatch` in the training loop; returns the
 * same shape. Per-iteration overhead is the worker spawn + net (de)serialise
 * (~1-2 s for a few processes), so the speedup is a bit under the worker count
 * (≈3× at 4 workers), not a perfect ×N.
 */

import { unlink } from "node:fs/promises";
import { snapshot, saveCheckpoint } from "./model-io";
import type { MLPPolicy } from "./policy";
import type { ConvValueNet } from "./value";
import type { AZSample, AZRolloutOpts, SelfPlayBatchResult } from "./alphazero";

export interface ParallelSelfPlayOpts extends AZRolloutOpts {
  games?: number;
  maxBatch?: number;
  terminalFraction?: number;
  workers?: number;
  tmpDir?: string;
}

export async function runParallelSelfPlay(
  policy: MLPPolicy, value: ConvValueNet, baseSeed: number, opts: ParallelSelfPlayOpts = {},
): Promise<SelfPlayBatchResult> {
  const workers = Math.max(1, opts.workers ?? 4);
  const games = opts.games ?? 32;
  const perWorker = Math.max(1, Math.round(games / workers));
  const tmp = opts.tmpDir ?? "/tmp";
  const stamp = `${process.pid}-${baseSeed}`;
  const ckptPath = `${tmp}/az-net-${stamp}.json`;
  const outPaths = Array.from({ length: workers }, (_, w) => `${tmp}/az-samples-${stamp}-${w}.json`);

  // Snapshot the current net for the workers to load.
  await saveCheckpoint(ckptPath, snapshot(policy, value, {}));

  const procs = outPaths.map((outPath, w) =>
    Bun.spawn(["bun", "run", "examples/selfplay-worker.ts"], {
      env: {
        ...process.env,
        WORKER_CKPT: ckptPath,
        WORKER_OUT: outPath,
        WORKER_GAMES: String(perWorker),
        WORKER_SIMS: String(opts.numSimulations ?? 60),
        WORKER_MAXPLIES: String(opts.maxPlies ?? 60),
        WORKER_SEED: String(baseSeed * 1000 + w + 1),
        WORKER_TEMPMOVES: String(opts.temperatureMoves ?? 24),
        WORKER_TERMFRAC: String(opts.terminalFraction ?? 0.25),
        WORKER_MAXBATCH: String(opts.maxBatch ?? 256),
      },
      stdout: "pipe",
      stderr: "pipe",
    }),
  );

  const codes = await Promise.all(procs.map((p) => p.exited));

  try {
    const samples: AZSample[] = [];
    let decisive = 0, plies = 0, gamesTotal = 0, batchSum = 0;
    for (let w = 0; w < workers; w++) {
      if (codes[w] !== 0) {
        const err = await new Response(procs[w]!.stderr).text();
        throw new Error(`self-play worker ${w} failed (exit ${codes[w]}):\n${err.slice(-800)}`);
      }
      const data = await Bun.file(outPaths[w]!).json() as {
        samples: { moveFeats: number[][]; planes: number[]; pi: number[]; z: number }[];
        decisive: number; games: number; avgPlies: number; avgBatch: number;
      };
      for (const s of data.samples) samples.push({ moveFeats: s.moveFeats, planes: new Float64Array(s.planes), pi: s.pi, z: s.z });
      decisive += data.decisive;
      gamesTotal += data.games;
      plies += data.avgPlies * data.games;
      batchSum += data.avgBatch;
    }
    return {
      samples, decisive, games: gamesTotal,
      avgPlies: gamesTotal ? plies / gamesTotal : 0,
      batches: 0, avgBatch: batchSum / workers,
    };
  } finally {
    // best-effort cleanup of temp files
    await Promise.allSettled([ckptPath, ...outPaths].map((p) => unlink(p)));
  }
}
