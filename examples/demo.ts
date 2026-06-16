/**
 * End-to-end wiring for steps 1–3: position → legal moves → feature vectors →
 * policy distribution over moves. Proves the whole pipeline runs on the Tensor
 * library before we add the self-play / REINFORCE loop (step 4).
 *
 * Run:  bun run examples/demo.ts
 */
import {
  parseFEN, startState, generateMoves, gameStatus, moveToUci,
  featureMatrix, FEATURE_DIM, FEATURE_NAMES,
  MLPPolicy, AttentionPolicy, type State,
} from "../src/index";

function topMoves(moves: ReturnType<typeof generateMoves>, probs: Float64Array, k = 6) {
  return [...moves.keys()]
    .sort((a, b) => probs[b]! - probs[a]!)
    .slice(0, k)
    .map((i) => `${moveToUci(moves[i]!).padEnd(6)} ${(probs[i]! * 100).toFixed(1)}%`);
}

async function show(label: string, s: State) {
  const moves = generateMoves(s);
  const feats = featureMatrix(s, moves);
  console.log(`\n=== ${label} ===`);
  console.log(`status: ${gameStatus(s)},  legal moves: ${moves.length},  feature dim: ${FEATURE_DIM}`);

  const mlp = new MLPPolicy({ seed: 7 });
  const attn = new AttentionPolicy({ seed: 7 });
  const out1 = await mlp.forward(feats);
  const out2 = await attn.forward(feats);

  console.log(`MLP  params: ${mlp.parameters().reduce((a, p) => a + p.view.length, 0)}`);
  console.log(`Attn params: ${attn.parameters().reduce((a, p) => a + p.view.length, 0)}`);
  console.log("untrained MLP  top moves:", topMoves(moves, out1.probs.view as Float64Array));
  console.log("untrained Attn top moves:", topMoves(moves, out2.probs.view as Float64Array));
  const sum = (out1.probs.view as Float64Array).reduce((a, b) => a + b, 0);
  console.log(`policy sums to ${sum.toFixed(6)} (should be 1)`);
}

const start = startState();
await show("start position", start);

// A sample feature vector, named, so the signals are legible.
const e4 = generateMoves(start).find((m) => moveToUci(m) === "e2e4")!;
const fv = featureMatrix(start, [e4])[0]!;
console.log("\nfeature vector for e2e4:");
console.log(FEATURE_NAMES.map((n, i) => `  ${n.padEnd(14)} ${fv[i]!.toFixed(3)}`).join("\n"));

// A tactical middlegame position (Kiwipete) — many captures/checks available.
await show("Kiwipete middlegame",
  parseFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));

process.exit(0);
