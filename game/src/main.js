import { createGameRuntime } from './game.js';
import { InputController } from './input.js';
import { CanvasRenderer } from './render.js';
import { UIController } from './ui.js';
import { GameAudio } from './audio.js';
import { loadAutosave } from './save.js';

const canvas=document.getElementById('game-canvas');
let runtime=createGameRuntime({seed:`ARES-${new Date().getUTCFullYear()}`});
const input=new InputController(window);const renderer=new CanvasRenderer(canvas);const audio=new GameAudio();let ui=new UIController(runtime,renderer,audio);
let pulseJump=false,last=performance.now(),acc=0;const STEP=1/60;

function applySettings(){
  const s=runtime.state.settings;
  document.documentElement.style.setProperty('--text-scale',String(s.textScale||1));
  document.body.classList.toggle('high-contrast',!!s.highContrast);
  input.sensitivity=s.controlSensitivity||1;
  audio.setVolume(s.volume??0.7);
  const values={volume:s.volume??0.7,'text-scale':s.textScale??1,'control-sensitivity':s.controlSensitivity??1,autosave:String(s.autosaveMinutes??3)};
  for(const [id,value] of Object.entries(values)){const el=document.getElementById(id);if(el)el.value=value;}
  const checks={'reduced-motion':!!s.reducedMotion,'high-contrast':!!s.highContrast};
  for(const [id,value] of Object.entries(checks)){const el=document.getElementById(id);if(el)el.checked=value;}
}
function bindSetting(id,key,kind='value'){
  document.getElementById(id)?.addEventListener('input',e=>{let v=kind==='checked'?e.target.checked:e.target.value;if(kind==='number')v=Number(v);runtime.state.settings[key]=v;applySettings();});
}
function wireUI(){
  ui.runtime=runtime;
  ui.onNewGame=seed=>{runtime=createGameRuntime({seed});ui.runtime=runtime;document.getElementById('title-screen').classList.add('hidden');audio.ensure();applySettings();};
  ui.onLoadState=state=>{runtime=createGameRuntime({state});ui.runtime=runtime;document.getElementById('title-screen').classList.add('hidden');ui.closePanel();applySettings();};
  ui.onPulseInput=name=>{if(name==='jump')pulseJump=true;};
}
wireUI();applySettings();

const hasSave=!!loadAutosave(localStorage);document.getElementById('continue-btn').disabled=!hasSave;
document.getElementById('mic-btn')?.addEventListener('click',async e=>{try{const ok=await audio.enableMic();e.currentTarget.textContent=ok?'MIC REACTIVE ✓':'MIC UNAVAILABLE';}catch{e.currentTarget.textContent='MIC DENIED';}});
bindSetting('volume','volume','number');bindSetting('text-scale','textScale','number');bindSetting('control-sensitivity','controlSensitivity','number');bindSetting('reduced-motion','reducedMotion','checked');bindSetting('high-contrast','highContrast','checked');bindSetting('autosave','autosaveMinutes','number');
input.bindVirtualStick(document.getElementById('stick'),document.getElementById('stick-knob'));

window.addEventListener('keydown',e=>{
  if(e.code==='KeyE')ui.action('gather'); if(e.code==='KeyF')ui.action('scan'); if(e.code==='KeyR')ui.action('rover');
  if(e.code==='KeyB')ui.openPanel('build'); if(e.code==='KeyI')ui.openPanel('inventory'); if(e.code==='KeyM')ui.openPanel('network');
  if(e.code==='Tab'){e.preventDefault();ui.openPanel('cst');} if(e.code==='Escape')runtime.state.flags.paused?ui.closePanel():ui.action('pause');
});
function frame(now){
  const dt=Math.min(.1,(now-last)/1000);last=now;acc+=dt;
  while(acc>=STEP){const snap=input.snapshot();if(pulseJump){snap.jump=true;pulseJump=false;}runtime.step(STEP,snap);acc-=STEP;}
  renderer.draw(runtime,{mic:audio.micLevel(),reducedMotion:runtime.state.settings.reducedMotion});ui.update();requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
