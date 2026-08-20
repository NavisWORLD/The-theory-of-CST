import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { generateWorld, sampleResource } from '../src/world.js';
test('same seed creates same initial state',()=>assert.deepEqual(createInitialState('ARES-01'),createInitialState('ARES-01')));
test('same world coordinate returns same resource sample',()=>{const a=generateWorld('ARES-01'),b=generateWorld('ARES-01');assert.deepEqual(sampleResource(a,120,-45),sampleResource(b,120,-45));});
test('new game carries accessibility and control settings',()=>{const s=createInitialState('SETTINGS');assert.equal(s.settings.highContrast,false);assert.equal(s.settings.reducedMotion,false);assert.equal(s.settings.screenShake,true);assert.equal(s.settings.controlSensitivity,1);assert.equal(s.settings.quality,'high');});
