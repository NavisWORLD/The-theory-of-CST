import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { updateCst, recordMemory, strengthenLink } from '../src/cst.js';

test('CST channels remain bounded', () => {
  const s = createInitialState('CST');
  s.inventory.ice = 99999;
  s.environment.dust = 1;
  updateCst(s, 1);
  assert.equal(s.cst.channels.length, 12);
  assert.ok(s.cst.channels.every(v => v >= 0 && v <= 1));
});

test('repeated cooperation strengthens one graph link', () => {
  const s = createInitialState('LINK');
  strengthenLink(s, 'hab-1', 'solar-1', 0.1);
  strengthenLink(s, 'hab-1', 'solar-1', 0.1);
  assert.equal(s.cst.links.length, 1);
  assert.ok(s.cst.links[0].weight > 0.1);
});

test('memory entries form a previous-id chain', () => {
  const s = createInitialState('MEM');
  recordMemory(s, 'first-water', 'Water located');
  recordMemory(s, 'storm', 'Storm survived');
  assert.equal(s.cst.memory[1].previousId, s.cst.memory[0].id);
  assert.notEqual(s.cst.memory[0].id, s.cst.memory[1].id);
});
