"use strict";

/**
 * Batched leaf evaluator for MCTS self-play.
 *
 * The generic MCTS engine evaluates one leaf at a time (sequential descent),
 * which is fine for a tiny scalar value net but wasteful for the conv value
 * net — every leaf is a separate conv forward. Since the engine's `evaluate`
 * may be async, we coalesce calls into batches WITHOUT touching the engine:
 *
 *   - Run G self-play games concurrently. Each game's search awaits its single
 *     pending leaf evaluation, so at any instant up to G requests are in flight.
 *   - `evaluate` enqueues (state, legal) and returns a promise. When the queue
 *     reaches the number of live games (or a microtask drains), we flush: ONE
 *     batched conv forward for all leaf values, ONE matmul for all leaf priors
 *     (moves of all leaves concatenated — they're scored independently), then
 *     resolve every pending promise.
 *
 * Result: the conv value net amortises over ~G positions per forward, which is
 * what makes a deeper search (more simulations) affordable. Inference only — no
 * gradients are needed here.
 */

import type { Evaluator, Evaluation } from "@euriklis/mcts";
import type { State, Move } from "./rules";
import { featureMatrix } from "./features";
import { repsOf } from "./mcts-player";
import type { MLPPolicy } from "./policy";
import type { ConvValueNet } from "./value";

interface Req { state: State; legal: Move[]; resolve: (e: Evaluation) => void; }

export class BatchedEvaluator {
  private queue: Req[] = [];
  private scheduled = false;
  /** Number of live games — set by the self-play driver so flushes coalesce. */
  active = 1;
  // telemetry
  batches = 0;
  positions = 0;

  constructor(
    private readonly policy: MLPPolicy,
    private readonly value: ConvValueNet,
    private readonly maxBatch = 256,
  ) {}

  /** The `Evaluator` to hand to `runMcts`. */
  evaluator(): Evaluator<State, Move> {
    return (state, legal) =>
      new Promise<Evaluation>((resolve) => {
        this.queue.push({ state, legal, resolve });
        const target = Math.max(1, Math.min(this.active, this.maxBatch));
        if (this.queue.length >= target) this.flush();
        else if (!this.scheduled) {
          // Defer the partial-batch flush to a MACROtask (setTimeout 0), not a
          // microtask: a microtask fires inside the current drain, before all
          // games have re-enqueued their next leaf, so batches collapse (and
          // how early depends on the Bun version's microtask scheduling — on
          // 1.3.x it coalesced only ~8/32). A macrotask fires AFTER the whole
          // microtask queue drains, so every game that will enqueue this round
          // has done so → near-full batches, deterministically across versions.
          this.scheduled = true;
          setTimeout(() => {
            this.scheduled = false;
            if (this.queue.length) this.flush();
          }, 0);
        }
      });
  }

  private flush(): void {
    const batch = this.queue;
    this.queue = [];
    void this.run(batch);
  }

  private async run(batch: Req[]): Promise<void> {
    this.batches++;
    this.positions += batch.length;

    // VALUE: one batched conv forward over all leaf positions → [B,1], each with
    // its repetition count (so leaves that repeat a position read as drawish).
    const vT = await this.value.forwardStates(batch.map((b) => b.state), batch.map((b) => repsOf(b.state)));
    const vv = vT.view as Float64Array;

    // POLICY: concatenate every leaf's legal-move features into one [ΣL, F]
    // matrix, score in one matmul, then softmax each leaf's segment.
    const feats: number[][] = [];
    const segLen: number[] = [];
    for (const b of batch) {
      const fm = featureMatrix(b.state, b.legal);
      segLen.push(fm.length);
      for (const row of fm) feats.push(row);
    }
    const scores = await this.policy.rawScores(feats); // [ΣL] (logits)

    let off = 0;
    for (let i = 0; i < batch.length; i++) {
      const L = segLen[i]!;
      // stable softmax over this leaf's moves
      let mx = -Infinity;
      for (let k = 0; k < L; k++) if (scores[off + k]! > mx) mx = scores[off + k]!;
      let z = 0;
      const priors = new Array<number>(L);
      for (let k = 0; k < L; k++) { const e = Math.exp(scores[off + k]! - mx); priors[k] = e; z += e; }
      for (let k = 0; k < L; k++) priors[k]! /= z;
      off += L;
      batch[i]!.resolve({ value: vv[i]!, priors });
    }
  }
}
