import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { evaluateCampaign, unlockResearch, updateProgression } from '../src/progression.js';

test('three linked stable settlements can trigger victory', () => {
  const s = createInitialState('WIN');
  s.settlements = [{id:'a',health:1},{id:'b',health:1},{id:'c',health:1}];
  s.cst.channels = Array(12).fill(0.86);
  s.cst.links = [{a:'a',b:'b',weight:0.9},{a:'b',b:'c',weight:0.9},{a:'a',b:'c',weight:0.9}];
  s.cst.entropy = 0.2;
  evaluateCampaign(s);
  assert.equal(s.flags.victory, true);
});

test('research unlock spends points once', () => {
  const s = createInitialState('TECH');
  s.research.points = 20;
  assert.equal(unlockResearch(s,'energy-1'), true);
  const after = s.research.points;
  assert.equal(unlockResearch(s,'energy-1'), false);
  assert.equal(s.research.points, after);
});

test('basic play advances touchdown mission chain', () => {
  const s = createInitialState('MISSION');
  s.structures.push({id:'solar-x',type:'solar',x:50,y:0,health:1});
  s.inventory.ice = 4;
  s.resources.power = 15;
  updateProgression(s);
  assert.ok(s.missions.completed.includes('restore-power'));
});
