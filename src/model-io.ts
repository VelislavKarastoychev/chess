"use strict";

/** Persist / restore trained weights (parameters are plain typed arrays). */

import { MLPPolicy } from "./policy";
import { ValueNet } from "./value";

export interface Checkpoint {
  policy: { hidden: number; params: number[][] };
  value: { hidden: number; params: number[][] };
  meta: Record<string, unknown>;
}

const dump = (net: { parameters(): { view: ArrayLike<number> }[] }): number[][] =>
  net.parameters().map((p) => Array.from(p.view));

const load = (net: { parameters(): { view: { set(a: number[]): void } }[] }, params: number[][]): void =>
  net.parameters().forEach((p, i) => p.view.set(params[i]!));

export function snapshot(policy: MLPPolicy, value: ValueNet, meta: Record<string, unknown> = {}): Checkpoint {
  return {
    policy: { hidden: policy.hidden, params: dump(policy) },
    value: { hidden: value.hidden, params: dump(value) },
    meta,
  };
}

export function restore(ckpt: Checkpoint): { policy: MLPPolicy; value: ValueNet } {
  const policy = new MLPPolicy({ hidden: ckpt.policy.hidden });
  const value = new ValueNet({ hidden: ckpt.value.hidden });
  load(policy, ckpt.policy.params);
  load(value, ckpt.value.params);
  return { policy, value };
}

export async function saveCheckpoint(path: string, ckpt: Checkpoint): Promise<void> {
  await Bun.write(path, JSON.stringify(ckpt));
}

export async function loadCheckpoint(path: string): Promise<Checkpoint> {
  return (await Bun.file(path).json()) as Checkpoint;
}
