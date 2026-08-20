export class GameAudio {
  constructor(){ this.ctx=null; this.master=null; this.ambience=null; this.mic=null; this.micAnalyser=null; this.micData=null; }
  ensure(){
    if(this.ctx) return this.ctx;
    const AC=globalThis.AudioContext||globalThis.webkitAudioContext;
    if(!AC) return null;
    this.ctx=new AC();
    this.master=this.ctx.createGain();
    this.master.gain.value=0.55;
    this.master.connect(this.ctx.destination);
    this.ambience=this.ctx.createGain();
    this.ambience.gain.value=0.08;
    this.ambience.connect(this.master);
    return this.ctx;
  }
  setVolume(v){ if(this.ensure()) this.master.gain.value=Math.max(0,Math.min(1,v)); }
  tone(freq=440,duration=0.12,gain=0.06,type='sine'){
    const ctx=this.ensure(); if(!ctx) return;
    const osc=ctx.createOscillator(), g=ctx.createGain();
    osc.type=type; osc.frequency.value=freq; g.gain.setValueAtTime(gain,ctx.currentTime); g.gain.exponentialRampToValueAtTime(0.0001,ctx.currentTime+duration);
    osc.connect(g); g.connect(this.master); osc.start(); osc.stop(ctx.currentTime+duration);
  }
  event(name){
    const tones={gather:[320,0.08],scan:[760,0.14],build:[180,0.18],mission:[520,0.28],error:[90,0.18],victory:[880,0.7]};
    const [f,d]=tones[name]??[420,0.08]; this.tone(f,d,name==='victory'?0.11:0.05,name==='error'?'sawtooth':'sine');
  }
  async enableMic(){
    const ctx=this.ensure();
    if(!ctx||!navigator.mediaDevices?.getUserMedia) return false;
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    this.mic=ctx.createMediaStreamSource(stream);
    this.micAnalyser=ctx.createAnalyser(); this.micAnalyser.fftSize=256; this.micData=new Uint8Array(this.micAnalyser.fftSize);
    this.mic.connect(this.micAnalyser); return true;
  }
  micLevel(){
    if(!this.micAnalyser) return 0;
    this.micAnalyser.getByteTimeDomainData(this.micData);
    let sum=0; for(const v of this.micData){const n=(v-128)/128;sum+=n*n;} return Math.min(1,Math.sqrt(sum/this.micData.length)*3);
  }
}
