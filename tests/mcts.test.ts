import { test, expect } from "bun:test";
import { parseFEN, moveToUci } from "../src/rules";
import { MLPPolicy } from "../src/policy";
import { ConvValueNet } from "../src/value";
import { chooseMoveMcts } from "../src/mcts-player";

/**
 * Integration: MCTS must find a mate-in-one even with UNTRAINED nets, because
 * the mating move reaches a terminal worth +1 that dominates every other line.
 * This validates the env reward/terminal wiring end-to-end through the search.
 */
test("MCTS finds mate-in-one (Ra8#)", async () => {
  const s = parseFEN("7k/5ppp/8/8/8/8/8/R6K w - - 0 1"); // Ra1→a8 is checkmate
  const policy = new MLPPolicy({ seed: 1 });
  const value = new ConvValueNet({ seed: 1 });
  const { move } = await chooseMoveMcts(s, policy, value, { numSimulations: 160 });
  expect(moveToUci(move)).toBe("a1a8");
});
