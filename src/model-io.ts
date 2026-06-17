"use strict";

/** Persist / restore trained weights (parameters are plain typed arrays). */

import { MLPPolicy } from "./policy";
import { ConvValueNet } from "./value";

export interface Checkpoint {
  policy: { hidden: number; params: number[][] };
  value: { kind: "conv"; channels: number; hidden: number; params: number[][] };
  meta: Record<string, unknown>;
}

const dump = (net: { parameters(): { view: ArrayLike<number> }[] }): number[][] =>
  net.parameters().map((p) => Array.from(p.view));

const load = (net: { parameters(): { view: { set(a: number[]): void } }[] }, params: number[][]): void =>
  net.parameters().forEach((p, i) => p.view.set(params[i]!));

export function snapshot(policy: MLPPolicy, value: ConvValueNet, meta: Record<string, unknown> = {}): Checkpoint {
  return {
    policy: { hidden: policy.hidden, params: dump(policy) },
    value: { kind: "conv", channels: value.channels, hidden: value.hidden, params: dump(value) },
    meta,
  };
}

/**
 * Rebuild policy + conv value net from a checkpoint. Migration-tolerant: a
 * compatible MLP policy is always restored; the conv value net is loaded only
 * if the checkpoint already holds conv weights (the old scalar value net is
 * incompatible, so it starts fresh and re-learns from the board planes).
 */
export function restore(ckpt: Checkpoint): { policy: MLPPolicy; value: ConvValueNet; freshValue: boolean } {
  const policy = new MLPPolicy({ hidden: ckpt.policy.hidden });
  load(policy, ckpt.policy.params);
  const v = ckpt.value;
  const isConv = v && (v as { kind?: string }).kind === "conv";
  const value = new ConvValueNet({ channels: isConv ? v.channels : undefined, hidden: isConv ? v.hidden : undefined });
  // Load only if EVERY param matches in element count too — the input plane
  // count is baked into the first conv weight, so an old (e.g. 18-plane)
  // checkpoint is shape-incompatible with a newer encoder and must start fresh.
  const params = value.parameters();
  const shapesMatch = isConv && Array.isArray(v.params) && v.params.length === params.length
    && params.every((p, i) => v.params[i]?.length === p.view.length);
  let freshValue = true;
  if (shapesMatch) {
    load(value, v.params);
    freshValue = false;
  }
  return { policy, value, freshValue };
}

export async function saveCheckpoint(path: string, ckpt: Checkpoint): Promise<void> {
  await Bun.write(path, JSON.stringify(ckpt));
}

export async function loadCheckpoint(path: string): Promise<Checkpoint> {
  return (await Bun.file(path).json()) as Checkpoint;
}
