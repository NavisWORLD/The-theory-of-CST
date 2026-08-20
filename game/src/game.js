import { createInitialState } from './state.js';
import { generateWorld, updateEnvironment, scanResources } from './world.js';
import { stepPlayer, stepSurvival, gatherAtPlayer, toggleVehicle } from './player.js';
import { stepProduction, placeStructure, createSettlement, craft } from './economy.js';
import { updateCst, recordMemory, strengthenLink } from './cst.js';
import { updateProgression, unlockResearch } from './progression.js';

export function createGameRuntime({ seed='ARES-01', state=null, headless=false }={}) {
  const runtime = {
    state: state ?? createInitialState(seed),
    world: null,
    headless,
    _networkClock:0,
    _lastStorm:null,
    _lastMission:null,
    step(dt=1/60,input={}) {
      if (this.state.flags.paused || this.state.flags.victory || this.state.flags.gameOver) return this.state;
      dt=Math.min(0.1,Math.max(0,dt));
      const oldStorm=this.state.environment.storm?.type ?? null;
      updateEnvironment(this.state,dt);
      stepPlayer(this.state,input,dt);
      stepSurvival(this.state,dt);
      stepProduction(this.state,dt);
      this._networkClock+=dt;
      if(this._networkClock>=1){
        this._networkClock=0;
        pulseLocalNetwork(this.state);
      }
      updateCst(this.state,dt);
      const beforeMission=this.state.missions.current;
      updateProgression(this.state);
      const newStorm=this.state.environment.storm?.type ?? null;
      if(oldStorm!==newStorm){
        if(newStorm) recordMemory(this.state,'hazard',`${newStorm==='dust'?'Dust storm':'Solar radiation event'} began`,{type:newStorm});
        else if(oldStorm) recordMemory(this.state,'hazard',`${oldStorm==='dust'?'Dust storm':'Solar radiation event'} cleared`,{type:oldStorm});
      }
      if(beforeMission!==this.state.missions.current) this._lastMission=this.state.missions.current;
      return this.state;
    },
    action(name,payload={}) {
      const s=this.state;
      if(name==='gather'){
        const result=gatherAtPlayer(s,this.world);
        if(result){
          if(result.type==='ice' && !s.cst.memory.some(m=>m.type==='first-water')) recordMemory(s,'first-water','First usable water ice located',{amount:result.amount});
          if(result.type==='anomaly') s.research.points+=2.5;
        }
        return result;
      }
      if(name==='toggle-rover') return toggleVehicle(s);
      if(name==='scan'){
        s.stats.scans+=1;
        s.research.points+=0.4;
        const samples=scanResources(this.world,s.player.x,s.player.y,220);
        for(const anomaly of this.world.anomalies){
          if(!anomaly.discovered && Math.hypot(anomaly.x-s.player.x,anomaly.y-s.player.y)<260){
            anomaly.discovered=true;
            s.inventory.anomaly+=1;
            s.research.points+=4;
            recordMemory(s,'anomaly','Buried anomaly signature resolved',{id:anomaly.id});
          }
        }
        return samples;
      }
      if(name==='build'){
        const angle=Number.isFinite(payload.angle)?payload.angle:s.player.facing;
        const distance=payload.distance??58;
        const x=payload.x??s.player.x+Math.cos(angle)*distance;
        const y=payload.y??s.player.y+Math.sin(angle)*distance;
        const ok=placeStructure(s,payload.type,x,y);
        if(ok && payload.type==='settlement-core') createSettlement(s,x,y,payload.name);
        return ok;
      }
      if(name==='research') return unlockResearch(s,payload.id);
      if(name==='craft') return craft(s,payload.recipe);
      if(name==='pause'){ s.flags.paused=!s.flags.paused; return s.flags.paused; }
      return null;
    },
    replaceState(nextState){ this.state=nextState; this.world=generateWorld(nextState.seed); this._lastStorm=nextState.environment.storm?.type??null; return this; }
  };
  runtime.world=generateWorld(runtime.state.seed);
  if(!state && runtime.state.cst.memory.length===0) recordMemory(runtime.state,'landing','Ares landing package touched down on Mars',{seed:runtime.state.seed});
  updateCst(runtime.state,1);
  updateProgression(runtime.state);
  return runtime;
}

function pulseLocalNetwork(state){
  const nodes=[...state.structures,...state.settlements];
  for(let i=0;i<nodes.length;i++){
    for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j];
      const d=Math.hypot((a.x??0)-(b.x??0),(a.y??0)-(b.y??0));
      const relay=a.type==='relay'||a.type==='cst-relay'||b.type==='relay'||b.type==='cst-relay';
      const range=relay?420:150;
      if(d<=range) strengthenLink(state,a.id,b.id,relay?0.003:0.0012);
    }
  }
}
