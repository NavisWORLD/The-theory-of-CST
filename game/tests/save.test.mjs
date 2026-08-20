import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { serializeGame, deserializeGame, saveSlot, loadSlot } from '../src/save.js';

test('save round trip preserves game state', () => {
  const s = createInitialState('SAVE');
  s.inventory.ice = 42;
  const restored = deserializeGame(serializeGame(s));
  assert.equal(restored.inventory.ice, 42);
  assert.equal(restored.seed, 'SAVE');
});

test('corrupt save returns null instead of throwing', () => {
  assert.equal(deserializeGame('{not-json'), null);
});

test('slot storage writes and loads a summary-compatible save', () => {
  const map = new Map();
  const storage = { setItem:(k,v)=>map.set(k,v), getItem:k=>map.get(k)??null, removeItem:k=>map.delete(k) };
  const s = createInitialState('SLOT');
  saveSlot(storage,s,2);
  assert.equal(loadSlot(storage,2).seed,'SLOT');
});
