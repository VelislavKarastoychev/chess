/**
 * Stateful "one move" driver so a human can play the trained net turn-by-turn
 * (e.g. relayed through a chat). Persists the game to game.json between calls.
 *
 *   bun run examples/move.ts new            # start, human = White (you move first)
 *   bun run examples/move.ts new black      # start, human = Black (engine moves first)
 *   bun run examples/move.ts e2e4           # play your move; engine replies
 *
 * Prints the board, the engine's reply, and the game status after each call.
 */
import { loadCheckpoint, restore } from "../src/model-io";
import { mulberry32 } from "../src/selfplay";
import {
  parseFEN, toFEN, startState, generateMoves, makeMove, gameStatus,
  moveToUci, parseUci, renderBoard, inCheck,
  WHITE, BLACK, type Color, type State,
} from "../src/rules";
import { whiteMaterial } from "../src/value";
import { chooseMoveMcts } from "../src/mcts-player";

const GAME = "game.json";
const CKPT = process.env.CHESS_CKPT ?? "checkpoints/best.json";
const SIMS = Number(process.env.CHESS_MCTS_SIMS ?? 200); // MCTS lookahead depth; 0 = raw net
const arg = (process.argv[2] ?? "").toLowerCase();

const { policy, value } = restore(await loadCheckpoint(CKPT));
const rng = mulberry32((Date.now() & 0x7fffffff) >>> 0);

async function engineMove(s: State) {
  const { move, search } = await chooseMoveMcts(s, policy, value, { numSimulations: SIMS, temperature: 0, rng });
  const top = [...search.ranking].sort((a, b) => b.visits - a.visits).slice(0, 3)
    .map((r) => `${moveToUci(r.action)}(${r.visits})`).join(" ");
  console.log(`  [MCTS ${SIMS} sims — top: ${top}]`);
  return move;
}

function report(s: State, human: Color) {
  console.log("\n" + renderBoard(s));
  const st = gameStatus(s);
  const mat = human * whiteMaterial(s.board);
  console.log(`\nstatus: ${st}${inCheck(s) ? " (check)" : ""}   material (your view): ${mat > 0 ? "+" : ""}${mat}`);
  if (st === "ongoing") console.log(`${s.turn === human ? "Your" : "Engine's"} move.`);
  else console.log(st === "checkmate" ? (s.turn === human ? "You are mated. Engine wins." : "Engine is mated. You win!") : "Draw.");
}

async function save(s: State, human: Color) {
  await Bun.write(GAME, JSON.stringify({ fen: toFEN(s), human: human === WHITE ? "white" : "black" }));
}

// ---- new game ----
if (arg === "new") {
  const human: Color = (process.argv[3] ?? "white").toLowerCase() === "black" ? BLACK : WHITE;
  const s = startState();
  console.log(`New game. You are ${human === WHITE ? "White (UPPERCASE)" : "Black (lowercase)"}.`);
  if (s.turn !== human) { const mv = await engineMove(s); console.log(`Engine plays: ${moveToUci(mv)}`); makeMove(s, mv); }
  report(s, human);
  await save(s, human);
  process.exit(0);
}

// ---- continue: apply human move, engine replies ----
if (!(await Bun.file(GAME).exists())) {
  console.log('No game in progress. Start one with:  bun run examples/move.ts new');
  process.exit(1);
}
const saved = await Bun.file(GAME).json() as { fen: string; human: string };
const human: Color = saved.human === "black" ? BLACK : WHITE;
const s = parseFEN(saved.fen);

if (s.turn !== human) { console.log("It's the engine's move — run with a move to let it reply."); }
const mv = parseUci(s, arg);
if (!mv) {
  console.log(`Illegal/unknown move "${arg}". Legal moves:`);
  console.log(generateMoves(s).map(moveToUci).join(" "));
  process.exit(1);
}
console.log(`You played: ${moveToUci(mv)}`);
makeMove(s, mv);

if (gameStatus(s) === "ongoing") {
  const reply = await engineMove(s);
  console.log(`Engine plays: ${moveToUci(reply)}`);
  makeMove(s, reply);
}
report(s, human);
await save(s, human);
process.exit(0);
