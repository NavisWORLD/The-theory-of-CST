import { TECHS } from './constants.js';
import { recordMemory } from './cst.js';

const MISSIONS = [
  { id:'restore-power', act:1, label:'Restore lander power', done:s=>s.resources.power>=8 && s.structures.some(x=>x.type==='solar') },
  { id:'locate-ice', act:1, label:'Locate usable water ice', done:s=>s.inventory.ice>=3 },
  { id:'safe-night', act:1, label:'Establish a safe night reserve', done:s=>s.resources.oxygen>=45 && s.resources.power>=8 },
  { id:'research-station', act:2, label:'Build a research station', done:s=>s.structures.some(x=>x.type==='research') },
  { id:'first-relay', act:2, label:'Deploy the first relay', done:s=>s.structures.some(x=>['relay','cst-relay'].includes(x.type)) },
  { id:'storm-survivor', act:2, label:'Survive a major storm', done:s=>s.stats.stormsSurvived>=1 },
  { id:'second-settlement', act:3, label:'Establish a second settlement', done:s=>s.settlements.length>=2 },
  { id:'network-two', act:3, label:'Link two settlements', done:s=>s.settlements.length>=2 && s.cst.links.some(l=>l.weight>=0.3) },
  { id:'third-settlement', act:4, label:'Establish a third settlement', done:s=>s.settlements.length>=3 },
  { id:'memory-archive', act:4, label:'Build a memory archive', done:s=>s.structures.some(x=>x.type==='memory-archive') },
  { id:'red-network', act:4, label:'Raise network coupling above 65%', done:s=>s.cst.channels[10]>=0.65 },
  { id:'genesis-threshold', act:5, label:'Stabilize the twelve-channel planetary field', done:s=>s.cst.channels.every(v=>v>=0.72) && s.cst.entropy<=0.35 },
];

export function completeMission(state, id) {
  if (state.missions.completed.includes(id)) return false;
  const mission = MISSIONS.find(m=>m.id===id);
  if (!mission) return false;
  state.missions.completed.push(id);
  state.research.points += 4 + mission.act * 2;
  state.missions.act = Math.max(state.missions.act, Math.min(5, mission.act));
  recordMemory(state,'mission',`Mission complete: ${mission.label}`,{id,act:mission.act});
  return true;
}

export function getCurrentMission(state) {
  return MISSIONS.find(m=>!state.missions.completed.includes(m.id)) ?? null;
}

export function unlockResearch(state, id) {
  const tech = TECHS[id];
  if (!tech || state.research.unlocked.includes(id)) return false;
  if (tech.requires && !state.research.unlocked.includes(tech.requires)) return false;
  if (state.research.points < tech.cost) return false;
  state.research.points -= tech.cost;
  state.research.unlocked.push(id);
  recordMemory(state,'research',`Research unlocked: ${tech.label}`,{id});
  return true;
}

function grantAchievement(state,id) {
  if (state.achievements.includes(id)) return false;
  state.achievements.push(id);
  recordMemory(state,'achievement',`Achievement unlocked: ${id}`,{id});
  return true;
}

export function updateAchievements(state) {
  if (state.resources.oxygen>80) grantAchievement(state,'first-breath');
  if (state.resources.water>20) grantAchievement(state,'red-water');
  if (state.stats.distance>2500) grantAchievement(state,'long-way-home');
  if (state.stats.stormsSurvived>=1) grantAchievement(state,'weathered');
  if (state.settlements.length>=2) grantAchievement(state,'two-nodes');
  if (state.settlements.length>=3) grantAchievement(state,'red-network');
  if (state.cst.memory.length>=25) grantAchievement(state,'remember-us');
}

export function updateProgression(state) {
  for (const mission of MISSIONS) if (!state.missions.completed.includes(mission.id) && mission.done(state)) completeMission(state,mission.id);
  const current=getCurrentMission(state);
  state.missions.current=current?.id ?? null;
  state.missions.act=current?.act ?? 5;
  updateAchievements(state);
  return evaluateCampaign(state);
}

export function evaluateCampaign(state) {
  const settlementIds=new Set(state.settlements.map(s=>s.id));
  const settlementLinks=state.cst.links.filter(l=>settlementIds.has(l.a)&&settlementIds.has(l.b)&&l.weight>=0.55);
  const avgChannel=state.cst.channels.reduce((a,b)=>a+b,0)/12;
  const strongChannels=state.cst.channels.every(v=>v>=0.78);
  const stable=state.cst.entropy<=0.32;
  const networkReady=state.settlements.length>=3 && settlementLinks.length>=2;
  if (networkReady && strongChannels && stable && avgChannel>=0.8) {
    if (!state.flags.victory) recordMemory(state,'genesis','Red Genesis planetary synapse activated',{avgChannel,entropy:state.cst.entropy});
    state.flags.victory=true;
    state.flags.gameOver=false;
  }
  const livingCore=state.settlements.some(settlement=>state.structures.some(s=>s.settlementId===settlement.id && ['habitat','settlement-core'].includes(s.type) && (s.health??1)>0.05));
  if (!livingCore && state.player.health<=0 && state.resources.oxygen<=0 && state.resources.power<=0) state.flags.gameOver=true;
  return state.flags.victory ? 'victory' : state.flags.gameOver ? 'failure' : 'playing';
}

export { MISSIONS };
