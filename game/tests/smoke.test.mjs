import test from 'node:test';
import assert from 'node:assert/strict';
import { createGameRuntime } from '../src/game.js';

test('runtime can advance a new game without browser globals', () => {
  const runtime = createGameRuntime({ seed:'SMOKE', headless:true });
  runtime.step(1/60, {});
  assert.equal(runtime.state.time.sol,1);
  assert.equal(runtime.state.cst.channels.length,12);
  assert.ok(runtime.world.rocks.length>50);
});

test('runtime actions connect gathering and build systems', () => {
  const runtime = createGameRuntime({ seed:'ACTION', headless:true });
  const before=runtime.state.stats.gathered;
  runtime.action('gather');
  assert.ok(runtime.state.stats.gathered>before);
});
