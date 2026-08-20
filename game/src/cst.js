import { hashString } from './random.js';

const clamp01 = v => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
const avg = xs => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0;

export function strengthenLink(state, a, b, amount = 0.05) {
  if (!a || !b || a === b) return null;
  const [x,y] = String(a) < String(b) ? [String(a),String(b)] : [String(b),String(a)];
  let link = state.cst.links.find(l => l.a === x && l.b === y);
  if (!link) {
    link = { a:x, b:y, weight:0.05, exchanges:0 };
    state.cst.links.push(link);
  }
  link.weight = clamp01(link.weight + Math.max(0, amount) * (1 - link.weight * 0.35));
  link.exchanges += 1;
  return link;
}

export function recordMemory(state, type, message, data = {}) {
  const previous = state.cst.memory.at(-1)?.id || null;
  const index = state.cst.memory.length;
  const payload = JSON.stringify({ seed:state.seed, index, type, message, data, previous });
  const id = `${hashString(payload).toString(16).padStart(8,'0')}-${index.toString(36)}`;
  const entry = {
    id, previousId:previous, index, sol:state.time.sol, elapsed:Math.round(state.time.elapsed*1000)/1000,
    type:String(type), message:String(message), data
  };
  state.cst.memory.push(entry);
  if (state.cst.memory.length > 300) state.cst.memory.splice(0, state.cst.memory.length - 300);
  return entry;
}

function structureHealth(state) {
  return state.structures.length ? avg(state.structures.map(s => clamp01(s.health ?? 1))) : 0;
}

function networkStats(state) {
  const connectedWeight = state.cst.links.reduce((sum,l)=>sum+clamp01(l.weight),0);
  const possible = Math.max(1, state.structures.length + state.settlements.length - 1);
  return { coupling:clamp01(connectedWeight/possible), connectedWeight };
}

export function computeCstChannels(state) {
  const active = type => state.structures.filter(s => s.type === type && (s.health ?? 1) > 0.15).length;
  const power = clamp01((state.resources.power + active('solar')*5 + active('reactor')*18) / Math.max(20, state.resources.powerCapacity || 20));
  const matter = clamp01((state.inventory.regolith + state.inventory.metal*2 + state.inventory.silica + state.inventory.salvage*2) / 180);
  const water = clamp01((state.resources.water + state.inventory.ice*0.35 + active('water')*8 + active('deep-drill')*12) / 100);
  const atmosphere = clamp01((state.resources.oxygen + active('oxygen')*10 + active('habitat')*7) / 120);
  const thermal = clamp01(0.34 + active('radiator')*0.13 + active('habitat')*0.05 - state.player.temperatureStress/150);
  const radiation = clamp01(0.35 + active('shelter')*0.14 + active('habitat')*0.05 - state.player.radiation/140);
  const mobility = clamp01((state.player.rover.health/100)*0.46 + (state.player.rover.battery/100)*0.24 + active('rover-bay')*0.18 + Math.min(0.12,state.stats.distance/18000));
  const knowledge = clamp01(state.research.points/75 + state.research.unlocked.length/18 + state.stats.scans/80 + state.inventory.anomaly/20);
  const memory = clamp01(state.cst.memory.length/60 + active('memory-archive')*0.18);
  const coherence = clamp01(structureHealth(state)*0.52 + state.settlements.length/6 + Math.min(0.18,state.stats.stormsSurvived/12));
  const { coupling } = networkStats(state);
  const adaptation = clamp01(state.research.unlocked.length/14 + Math.min(0.28,state.stats.stormsSurvived/10) + Math.min(0.2,state.stats.rescues/8) + active('lattice-hub')*0.16);
  return [power,matter,water,atmosphere,thermal,radiation,mobility,knowledge,memory,coherence,coupling,adaptation];
}

export function updateCst(state, dt = 1/60) {
  const target = computeCstChannels(state);
  const alpha = clamp01(dt * 0.45);
  for (let i=0;i<12;i++) {
    const left = state.cst.channels[(i+11)%12] ?? 0;
    const right = state.cst.channels[(i+1)%12] ?? 0;
    const coupledTarget = target[i]*0.82 + ((left+right)/2)*0.18;
    state.cst.channels[i] = clamp01(state.cst.channels[i] + (coupledTarget-state.cst.channels[i])*alpha);
  }

  for (const link of state.cst.links) link.weight = clamp01(link.weight - 0.000015 * dt);
  const damage = 1 - structureHealth(state);
  const reserveRisk = clamp01((25-state.resources.oxygen)/25) * 0.35 + clamp01((8-state.resources.power)/8) * 0.25;
  const hazard = clamp01(state.environment.dust*0.42 + state.environment.radiation*0.26 + (state.environment.storm ? state.environment.storm.severity*0.32 : 0));
  const disconnect = clamp01(1 - state.cst.channels[10]);
  const variation = avg(state.cst.channels.map((v,i)=>Math.abs(v-target[i])));
  const targetEntropy = clamp01(hazard*0.46 + damage*0.2 + reserveRisk*0.14 + disconnect*0.12 + variation*0.08);
  state.cst.entropy = clamp01(state.cst.entropy + (targetEntropy-state.cst.entropy)*Math.min(1,dt*0.22));

  const energy = state.cst.channels[0];
  const coupling = state.cst.channels[10];
  const memory = state.cst.channels[8];
  const adaptation = state.cst.channels[11];
  const instabilityOpportunity = state.cst.entropy * 0.16;
  state.cst.psiProxy = clamp01(energy*0.28 + coupling*0.28 + memory*0.18 + adaptation*0.18 + instabilityOpportunity - state.cst.entropy*0.08);
  return state.cst;
}
