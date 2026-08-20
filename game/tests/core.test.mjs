import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { generateWorld, sampleResource } from '../src/world.js';

test('same seed creates same initial state', () => {
  assert.deepEqual(createInitialState('ARES-01'), createInitialState('ARES-01'));
});

test('same world coordinate returns same resource sample', () => {
  const a = generateWorld('ARES-01');
  const b = generateWorld('ARES-01');
  assert.deepEqual(sampleResource(a, 120, -45), sampleResource(b, 120, -45));
});
