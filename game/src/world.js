import { createRng, seededValue } from './random.js';
import { RESOURCE_TYPES, SOL_SECONDS, WORLD_RADIUS } from './constants.js';

export function generateWorld(seed) {
  const rng = createRng(`world:${seed}`);
  const rocks = [];
  for (let i=0;i<180;i++) {
    const a = rng()*Math.PI*2;
    const r = 60 + rng()*(WORLD_RADIUS-80);
    rocks.push({ x:Math.cos(a)*r, y:Math.sin(a)*r, radius:3+rng()*12, shade:rng() });
  }
  const anomalies = [];
  for (let i=0;i<9;i++) {
    const a = rng()*Math.PI*2;
    const r = 220 + rng()*1800;
    anomalies.push({ id:`anomaly-${i}`, x:Math.cos(a)*r, y:Math.sin(a)*r, discovered:false });
  }
  return { seed:String(seed), radius:WORLD_RADIUS, rocks, anomalies };
}

export function terrainHeight(world, x, y) {
  const a = seededValue(world.seed, Math.floor(x/70), Math.floor(y/70), 'terrain');
  const b = seededValue(world.seed, Math.floor(x/25), Math.floor(y/25), 'detail');
  return (a*0.75 + b*0.25 - 0.5) * 22;
}

export function sampleResource(world, x, y) {
  const gx = Math.floor(x/34), gy = Math.floor(y/34);
  const weights = RESOURCE_TYPES.map((type, i) => ({ type, v: seededValue(world.seed, gx, gy, `resource:${type}:${i}`) }));
  weights.sort((a,b)=>b.v-a.v);
  const top = weights[0];
  const richness = Math.max(0.08, Math.min(1, (top.v - 0.5) * 1.7 + 0.25));
  return { type: top.type, richness, amount: Math.max(1, Math.round(1 + richness*5)) };
}

export function scanResources(world, x, y, radius=180) {
  const samples = [];
  for (let dx=-radius; dx<=radius; dx+=90) for (let dy=-radius; dy<=radius; dy+=90) {
    if (dx*dx+dy*dy <= radius*radius) samples.push({ x:x+dx,y:y+dy,...sampleResource(world,x+dx,y+dy) });
  }
  return samples.sort((a,b)=>b.richness-a.richness).slice(0,8);
}

export function updateEnvironment(state, dt) {
  state.time.elapsed += dt;
  state.stats.playtime += dt;
  state.time.sol = Math.floor(state.time.elapsed / SOL_SECONDS) + 1;
  state.time.solPhase = (state.time.elapsed % SOL_SECONDS) / SOL_SECONDS;
  const sunAngle = state.time.solPhase * Math.PI * 2;
  state.environment.solar = Math.max(0, Math.sin(sunAngle) * 0.92) * (1 - state.environment.dust * 0.72);
  state.environment.temperature = -82 + Math.max(0, Math.sin(sunAngle)) * 58 - state.environment.dust * 9;
  state.environment.radiation = 0.12 + (1-state.environment.dust)*0.19 + (state.environment.storm?.type === 'solar' ? 0.52 : 0);
  state.environment.stormClock -= dt;
  if (state.environment.storm && state.environment.stormClock <= 0) {
    if (state.environment.storm.type === 'dust') state.stats.stormsSurvived += 1;
    state.environment.storm = null;
    state.environment.dust = Math.max(0.1, state.environment.dust * 0.62);
    state.environment.stormClock = 70 + seededValue(state.seed, state.time.sol, Math.floor(state.time.elapsed), 'nextStorm') * 130;
  } else if (!state.environment.storm && state.environment.stormClock <= 0) {
    const v = seededValue(state.seed, state.time.sol, Math.floor(state.time.elapsed/30), 'storm');
    state.environment.storm = { type: v > 0.8 ? 'solar' : 'dust', severity: 0.42 + v*0.5 };
    state.environment.dust = state.environment.storm.type === 'dust' ? Math.min(1, 0.55 + v*0.4) : state.environment.dust;
    state.environment.stormClock = 35 + v*55;
  }
}
