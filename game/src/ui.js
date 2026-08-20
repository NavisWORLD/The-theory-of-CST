import { BUILDINGS, TECHS } from './constants.js';
import { MISSIONS } from './progression.js';
import { saveSlot, loadSlot, getSlotSummary, saveAutosave, loadAutosave } from './save.js';

const NAMES=['Energy','Matter','Water','Atmosphere','Thermal','Radiation','Mobility','Knowledge','Memory','Coherence','Network','Adaptation'];
const pct=v=>`${Math.round(Math.max(0,Math.min(100,v)))}%`;
const fmt=v=>Number(v??0).toFixed(v>=10?0:1);

export class UIController {
  constructor(runtime,renderer,audio){this.runtime=runtime;this.renderer=renderer;this.audio=audio;this.root=document;this.activePanel=null;this.lastAutosave=0;this.bind();this.buildStaticPanels();}
  bind(){
    document.querySelectorAll('[data-action]').forEach(btn=>btn.addEventListener('click',()=>this.action(btn.dataset.action,btn.dataset.value)));
    document.getElementById('seed-form')?.addEventListener('submit',e=>{e.preventDefault();this.onNewGame?.(document.getElementById('seed-input').value.trim()||`ARES-${Date.now().toString(36).slice(-5).toUpperCase()}`);});
    document.getElementById('continue-btn')?.addEventListener('click',()=>{const s=loadAutosave(localStorage)||loadSlot(localStorage,1);if(s)this.onLoadState?.(s);});
    document.getElementById('panel-close')?.addEventListener('click',()=>this.closePanel());
  }
  buildStaticPanels(){
    const build=document.getElementById('build-list');
    if(build) build.innerHTML=Object.entries(BUILDINGS).map(([id,b])=>`<button class="card-button" data-build="${id}"><strong>${b.label}</strong><small>${Object.entries(b.cost).map(([k,v])=>`${k} ${v}`).join(' · ')}</small></button>`).join('');
    build?.querySelectorAll('[data-build]').forEach(b=>b.addEventListener('click',()=>{const ok=this.runtime.action('build',{type:b.dataset.build});this.audio.event(ok?'build':'error');if(ok)this.closePanel();}));
    const tech=document.getElementById('research-list');
    if(tech) tech.innerHTML=Object.entries(TECHS).map(([id,t])=>`<button class="card-button" data-tech="${id}"><strong>${t.label}</strong><small>${t.cost} RP${t.requires?` · requires ${t.requires}`:''}</small></button>`).join('');
    tech?.querySelectorAll('[data-tech]').forEach(b=>b.addEventListener('click',()=>{const ok=this.runtime.action('research',{id:b.dataset.tech});this.audio.event(ok?'mission':'error');this.updatePanels();}));
    for(let slot=1;slot<=3;slot++) document.querySelector(`[data-save-slot="${slot}"]`)?.addEventListener('click',()=>{saveSlot(localStorage,this.runtime.state,slot);this.audio.event('mission');this.updatePanels();});
    for(let slot=1;slot<=3;slot++) document.querySelector(`[data-load-slot="${slot}"]`)?.addEventListener('click',()=>{const s=loadSlot(localStorage,slot);if(s)this.onLoadState?.(s);});
  }
  action(name,value){
    if(name==='gather'){const r=this.runtime.action('gather');this.audio.event(r?'gather':'error');}
    else if(name==='scan'){this.renderer.scanSamples=this.runtime.action('scan')||[];this.audio.event('scan');setTimeout(()=>this.renderer.scanSamples=[],8500);}
    else if(name==='rover'){this.runtime.action('toggle-rover');this.audio.event('build');}
    else if(name==='panel')this.openPanel(value);
    else if(name==='pause'){this.runtime.action('pause');this.openPanel('pause');}
    else if(name==='jump'){this.onPulseInput?.('jump');}
  }
  openPanel(name){this.activePanel=name;document.getElementById('panel')?.classList.remove('hidden');document.querySelectorAll('.panel-section').forEach(s=>s.classList.toggle('active',s.dataset.section===name));this.updatePanels();}
  closePanel(){this.activePanel=null;document.getElementById('panel')?.classList.add('hidden');if(this.runtime.state.flags.paused)this.runtime.state.flags.paused=false;}
  update(){
    const s=this.runtime.state,p=s.player;
    setMeter('health',p.health);setMeter('oxygen',p.oxygen);setMeter('energy',p.energy);setMeter('radiation',100-p.radiation);setMeter('temperature',100-p.temperatureStress);
    setText('sol',`SOL ${s.time.sol}`);setText('clock',`${String(Math.floor(s.time.solPhase*24)).padStart(2,'0')}:${String(Math.floor((s.time.solPhase*24%1)*60)).padStart(2,'0')}`);
    setText('mode',p.mode==='rover'?`ROVER ${Math.round(p.rover.battery)}%`:'EVA');
    setText('storm',s.environment.storm?`${s.environment.storm.type.toUpperCase()} EVENT`:`${Math.round(s.environment.temperature)}°C`);
    const mission=MISSIONS.find(m=>m.id===s.missions.current);setText('objective',mission?.label??(s.flags.victory?'RED GENESIS ONLINE':'Stabilize the network'));
    setText('psi',`ψ ${Math.round(s.cst.psiProxy*100)} · H ${Math.round(s.cst.entropy*100)}`);
    const ring=document.getElementById('mini-cst');if(ring)ring.style.setProperty('--cst',`${Math.round(s.cst.psiProxy*360)}deg`);
    if(s.flags.victory)document.getElementById('victory')?.classList.remove('hidden');
    if(s.flags.gameOver)document.getElementById('failure')?.classList.remove('hidden');
    const now=performance.now();if(now-this.lastAutosave>(s.settings.autosaveMinutes||3)*60000){saveAutosave(localStorage,s);this.lastAutosave=now;}
    if(this.activePanel)this.updatePanels();
  }
  updatePanels(){
    const s=this.runtime.state;
    const inv=document.getElementById('inventory-grid');if(inv)inv.innerHTML=[...Object.entries(s.inventory),...Object.entries(s.resources)].map(([k,v])=>`<div><span>${k}</span><strong>${fmt(v)}</strong></div>`).join('');
    const cst=document.getElementById('cst-grid');if(cst)cst.innerHTML=s.cst.channels.map((v,i)=>`<div class="cst-row"><span>D${String(i+1).padStart(2,'0')} ${NAMES[i]}</span><div><i style="width:${v*100}%"></i></div><b>${Math.round(v*100)}</b></div>`).join('')+`<p class="boundary">Gameplay proxy only. ψ and the 12 channels are simulation variables, not measured physical fields.</p>`;
    const mem=document.getElementById('memory-list');if(mem)mem.innerHTML=s.cst.memory.slice(-20).reverse().map(m=>`<article><b>${m.type}</b><span>${m.message}</span><small>Sol ${m.sol} · ${m.id}</small></article>`).join('')||'<p>No memories yet.</p>';
    const missions=document.getElementById('mission-list');if(missions)missions.innerHTML=MISSIONS.map(m=>`<div class="mission ${s.missions.completed.includes(m.id)?'done':''}"><b>Act ${m.act}</b><span>${m.label}</span></div>`).join('');
    const tech=document.getElementById('research-points');if(tech)tech.textContent=`${Math.floor(s.research.points)} research points`;
    document.querySelectorAll('[data-tech]').forEach(b=>b.classList.toggle('owned',s.research.unlocked.includes(b.dataset.tech)));
    const saves=document.getElementById('save-summaries');if(saves)saves.innerHTML=[1,2,3].map(slot=>{const x=getSlotSummary(localStorage,slot);return `<div>Slot ${slot}: ${x?`${x.seed} · Sol ${x.sol} · ${x.settlements} settlements`:'empty'}</div>`}).join('');
  }
}
function setText(id,text){const el=document.getElementById(id);if(el)el.textContent=text;}
function setMeter(id,value){const el=document.getElementById(`${id}-meter`);if(el)el.style.width=`${Math.max(0,Math.min(100,value))}%`;setText(`${id}-value`,pct(value));}
