/**
 * Self-play worker process — one of N spawned by the trainer each iteration to
 * generate games in parallel (the only way to use a multi-core box here, since
 * the MCTS/self-play loop is single-threaded JS). Loads the current net from a
 * checkpoint, runs `selfPlayBatch` for its share of games, and writes the
 * resulting samples (as JSON) to an output file the trainer collects.
 *
 * Parameters come via env (set by src/parallel-selfplay.ts):
 *   WORKER_CKPT, WORKER_OUT, WORKER_GAMES, WORKER_SIMS, WORKER_MAXPLIES,
 *   WORKER_SEED, WORKER_TEMPMOVES, WORKER_TERMFRAC, WORKER_MAXBATCH
 */
import { loadCheckpoint, restore } from "../src/model-io";
import { selfPlayBatch } from "../src/alphazero";
import { mulberry32 } from "../src/selfplay";

const ckptPath = process.env.WORKER_CKPT!;
const outPath = process.env.WORKER_OUT!;
const games = Number(process.env.WORKER_GAMES ?? 8);
const sims = Number(process.env.WORKER_SIMS ?? 60);
const maxPlies = Number(process.env.WORKER_MAXPLIES ?? 60);
const seed = Number(process.env.WORKER_SEED ?? 1);
const tempMoves = Number(process.env.WORKER_TEMPMOVES ?? 24);
const termFrac = Number(process.env.WORKER_TERMFRAC ?? 0.25);
const maxBatch = Number(process.env.WORKER_MAXBATCH ?? 256);

const { policy, value } = restore(await loadCheckpoint(ckptPath));
const sp = await selfPlayBatch(policy, value, mulberry32(seed), {
  games, numSimulations: sims, maxPlies, temperatureMoves: tempMoves,
  terminalFraction: termFrac, maxBatch, dirichletAlpha: 0.3, cPuct: 1.5,
});

// Float64Array doesn't survive JSON.stringify as an array — convert planes.
const samples = sp.samples.map((s) => ({ moveFeats: s.moveFeats, planes: Array.from(s.planes), pi: s.pi, z: s.z }));
await Bun.write(outPath, JSON.stringify({ samples, decisive: sp.decisive, games: sp.games, avgPlies: sp.avgPlies, avgBatch: sp.avgBatch }));
process.exit(0);
