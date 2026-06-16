/**
 * Play chess against the trained network in your terminal.
 *
 *   bun run examples/human-vs-net.ts                       # you are White
 *   bun run examples/human-vs-net.ts checkpoints/best.json black
 *
 * Enter moves in UCI: from-square + to-square (+ promotion piece), e.g.
 *   e2e4   g1f3   e1g1 (castle)   e7e8q (promote to queen).
 * Type "moves" to list legal moves, "board" to redraw, "quit" to exit.
 *
 * The engine samples with low temperature (some variety, mostly its best move).
 */
import { loadCheckpoint, restore } from "../src/model-io";
import { mulberry32, sampleFromProbs } from "../src/selfplay";
import {
  startState, generateMoves, makeMove, gameStatus, moveToUci, parseUci,
  renderBoard, inCheck, WHITE, BLACK, type Color, type State,
} from "../src/rules";
import { featureMatrix } from "../src/features";
import { type MLPPolicy } from "../src/policy";

const path = process.argv[2] ?? "checkpoints/best.json";
const humanColor: Color = process.argv[3]?.toLowerCase() === "black" ? BLACK : WHITE;
const TEMP = 0.4;

const { policy } = restore(await loadCheckpoint(path));
// Seed varies per run so the engine isn't identical every game.
const rng = mulberry32((Date.now() & 0x7fffffff) >>> 0);

async function engineMove(p: MLPPolicy, s: State) {
  const moves = generateMoves(s);
  const out = await p.forward(featureMatrix(s, moves));
  const logits = Array.from(out.scores.view as Float64Array).map((x) => x / TEMP);
  const max = Math.max(...logits);
  const ex = logits.map((x) => Math.exp(x - max));
  const z = ex.reduce((a, b) => a + b, 0);
  return moves[sampleFromProbs(ex.map((e) => e / z), rng)]!;
}

const show = (s: State) => {
  console.log("\n" + renderBoard(s));
  console.log(`\n${s.turn === WHITE ? "White" : "Black"} to move${inCheck(s) ? " — CHECK" : ""}`);
};

console.log(`Loaded ${path}. You are ${humanColor === WHITE ? "White (uppercase)" : "Black (lowercase)"}.`);
const s = startState();
show(s);

while (true) {
  const status = gameStatus(s);
  if (status !== "ongoing") {
    console.log(`\nGame over: ${status}.`);
    break;
  }

  if (s.turn === humanColor) {
    const input = prompt("\nYour move:")?.trim().toLowerCase() ?? "quit";
    if (input === "quit" || input === "q") { console.log("bye."); break; }
    if (input === "board") { show(s); continue; }
    if (input === "moves") { console.log(generateMoves(s).map(moveToUci).join(" ")); continue; }
    const mv = parseUci(s, input);
    if (!mv) { console.log(`illegal/unknown move "${input}". Type "moves" to see options.`); continue; }
    makeMove(s, mv);
    show(s);
  } else {
    const mv = await engineMove(policy, s);
    console.log(`\nEngine plays: ${moveToUci(mv)}`);
    makeMove(s, mv);
    show(s);
  }
}
process.exit(0);
