import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { placeStructure, stepProduction, canAfford, createSettlement } from '../src/economy.js';

test('placing a structure spends materials and creates a node', () => {
  const s = createInitialState('BUILD');
  Object.assign(s.inventory, { regolith: 50, metal: 50, silica: 50, salvage: 50, ice: 50, carbon: 50, anomaly: 10 });
  const before = s.inventory.metal;
  const ok = placeStructure(s, 'solar', 45, 0);
  assert.equal(ok, true);
  assert.equal(s.structures.length, 2);
  assert.ok(s.inventory.metal < before);
});

test('powered oxygen extractor raises oxygen reserve', () => {
  const s = createInitialState('PROD');
  s.resources.oxygen = 10;
  s.structures.push({ id:'solar-1', type:'solar', x:0, y:0, health:1, settlementId:'alpha' });
  s.structures.push({ id:'oxygen-1', type:'oxygen', x:5, y:0, health:1, settlementId:'alpha' });
  s.inventory.ice = 20;
  const before = s.resources.oxygen;
  stepProduction(s, 10);
  assert.ok(s.resources.oxygen > before);
});

test('cannot afford expensive structure without materials', () => {
  const s = createInitialState('POOR');
  for (const k of Object.keys(s.inventory)) s.inventory[k] = 0;
  assert.equal(canAfford(s, 'settlement-core'), false);
});

test('settlement core creates a settlement entry', () => {
  const s = createInitialState('SETTLE');
  Object.assign(s.inventory, { regolith: 100, metal: 100, silica: 100, salvage: 100, anomaly:10 });
  const core = placeStructure(s, 'settlement-core', 300, 0);
  assert.equal(core, true);
  const made = createSettlement(s, 300, 0, 'Second Light');
  assert.equal(made.name, 'Second Light');
  assert.equal(s.settlements.length, 2);
});
