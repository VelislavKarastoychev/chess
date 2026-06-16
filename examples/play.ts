/**
 * Load a trained checkpoint and let it play. Uses temperature SAMPLING (not
 * greedy argmax), which avoids the degenerate repetition loops that pure-greedy
 * play falls into. Our side plays the policy; the opponent is random.
 *
 * Run:  bun run examples/play.ts [checkpoints/best.json]
 */
import { loadCheckpoint, restore } from "../src/model-io";
import { mulberry32, sampleFromProbs } from "../src/selfplay";
import { startState, generateMoves, makeMove, gameStatus, moveToUci, inCheck, type Color } from "../src/rules";
import { featureMatrix } from "../src/features";
import { whiteMaterial } from "../src/value";

const path = process.argv[2] ?? "checkpoints/best.json";
const { policy } = restore(await loadCheckpoint(path));
const rng = mulberry32(42);
const TEMP = 0.6; // <1 sharpens toward better moves but keeps exploration

console.log(`loaded ${path} — policy plays White (sampled, temp ${TEMP}) vs random Black\n`);

const s = startState();
const ourColor: Color = 1;
const line: string[] = [];
for (let ply = 0; ply < 120; ply++) {
  const moves = generateMoves(s);
  if (moves.length === 0) break;
  if (s.halfmove >= 100) break;
  let idx: number;
  if (s.turn === ourColor) {
    const out = await policy.forward(featureMatrix(s, moves));
    const logits = Array.from(out.scores.view as Float64Array).map((x) => x / TEMP);
    const m = Math.max(...logits);
    const ex = logits.map((x) => Math.exp(x - m));
    const z = ex.reduce((a, b) => a + b, 0);
    idx = sampleFromProbs(ex.map((e) => e / z), rng);
  } else idx = Math.floor(rng() * moves.length);
  line.push((s.turn === ourColor ? "" : "") + moveToUci(moves[idx]!));
  makeMove(s, moves[idx]!);
}

console.log(line.join(" "));
console.log(`\nend: ${gameStatus(s)}${inCheck(s) ? " (check)" : ""}`);
console.log(`final material (our view): ${(ourColor * whiteMaterial(s.board)).toFixed(1)} pawns`);
process.exit(0);
