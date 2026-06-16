# @euriklis/chess

Feature-vector **move-scoring** chess policy on top of the `@euriklis/mathematics`
`Tensor` library. The plan: each *legal* move is described by a hand-designed
vector of tactical/positional signals; a policy network scores the set of
candidate moves into a distribution `π(a|s)`; training is reinforcement learning
via self-play. (Precedent: Giraffe, 2015 — handcrafted features + neural eval + RL.)

## Status

| Step | What | State |
|------|------|-------|
| 1 | `rules.ts` — board, legal move gen, make/unmake, FEN, `perft` | ✅ verified by perft (startpos, Kiwipete, pos 3 & 5) |
| 2 | `features.ts` — `moveFeatures(s, m)` → 23-dim vector | ✅ |
| 3 | `policy.ts` — `MLPPolicy` + `AttentionPolicy` over the move set | ✅ runs end-to-end |
| 4 | `value.ts` + `selfplay.ts` — self-play actor-critic (REINFORCE + value baseline) | ✅ trains, beats its own baseline |

## Try it

```bash
bun run examples/demo.ts   # position → legal moves → features → policy distribution
bun run examples/train.ts  # self-play actor-critic training vs a random opponent
bun test                   # perft correctness + REINFORCE mechanics
```

A 30-iteration self-play run (≈800 params) lifts average material vs a random
opponent from **−3.9 → +34 pawns** — the policy learns to win material and stop
hanging pieces. Win-rate stays near 50% because games are ply-capped and the
random side is rarely mated inside the cap; material is the honest progress
proxy at this scale.

## Design notes

- **Legality is free**: we only ever score moves from the legal generator — no
  illegal-move problem (the main weakness of the pure language-model approach).
- **Colour-symmetric features**: ranks are taken from the mover's perspective,
  so both colours share one set of weights.
- **Why the transformer**: `MLPPolicy` scores each move independently (baseline);
  `AttentionPolicy` runs a non-causal transformer block so moves attend to each
  other and are scored *relative to their alternatives*. Validate the MLP learns
  first, then measure what attention adds.

## Next (step 4)

REINFORCE-with-baseline loss is buildable from the available ops:
`logπ = log(softmax(scores))`, select the played move via a one-hot `times`,
`loss = -(advantage · logπ).mean()`, value head MSE for the baseline. Cold-start
RL on chess is sample-hungry — plan to shape reward with the handcrafted eval and
warm-start from a simple heuristic before pure self-play.
