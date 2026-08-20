import { GAME_SCHEMA_VERSION } from './constants.js';

export function createInitialState(seed = 'ARES-01') {
  return {
    schemaVersion: GAME_SCHEMA_VERSION,
    seed: String(seed),
    time: { elapsed: 0, sol: 1, solPhase: 0.24 },
    player: {
      x: 0, y: 0, vx: 0, vy: 0, altitude: 0, verticalVelocity: 0,
      facing: 0, health: 100, oxygen: 100, energy: 100, radiation: 0,
      temperatureStress: 0, mode: 'foot', gatherCooldown: 0, recoveryCount: 0,
      rover: { health: 100, battery: 100, cargo: 0, maxCargo: 120 }
    },
    inventory: { ice: 0, regolith: 12, metal: 10, silica: 8, carbon: 0, salvage: 6, anomaly: 0 },
    resources: { water: 8, oxygen: 72, power: 10, powerCapacity: 20 },
    structures: [{ id:'habitat-0', type:'habitat', x:0, y:0, health:1, settlementId:'alpha' }],
    settlements: [{ id:'alpha', name:'Ares Landing', x:0, y:0, health:1 }],
    research: { points: 0, unlocked: ['survival-1'] },
    missions: { act: 1, completed: [], current: 'restore-power' },
    achievements: [],
    cst: { channels: Array(12).fill(0.1), entropy: 0.2, psiProxy: 0, links: [], memory: [] },
    environment: { dust: 0.15, radiation: 0.18, temperature: -35, solar: 0.75, storm: null, stormClock: 0 },
    stats: { gathered: 0, built: 0, distance: 0, stormsSurvived: 0, scans: 0, rescues: 0, playtime: 0 },
    flags: { paused: false, gameOver: false, victory: false, tutorialDone: false },
    settings: { reducedMotion:false, screenShake:true, volume:0.7, music:0.45, effects:0.7, textScale:1, quality:'high', autosaveMinutes:3 }
  };
}

export function cloneState(state) {
  return structuredClone ? structuredClone(state) : JSON.parse(JSON.stringify(state));
}

export function isValidStateShape(value) {
  return !!value && value.schemaVersion === GAME_SCHEMA_VERSION && typeof value.seed === 'string' &&
    Array.isArray(value?.cst?.channels) && value.cst.channels.length === 12 && Array.isArray(value.structures);
}
