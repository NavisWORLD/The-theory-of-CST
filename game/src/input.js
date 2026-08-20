export class InputController {
  constructor(target = globalThis.window) {
    this.keys = new Set();
    this.axisX = 0;
    this.axisY = 0;
    this.sensitivity = 1;
    this.target = target;
    this._onDown = e => {
      this.keys.add(e.code);
      if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].includes(e.code)) e.preventDefault();
    };
    this._onUp = e => this.keys.delete(e.code);
    target?.addEventListener?.('keydown', this._onDown, { passive:false });
    target?.addEventListener?.('keyup', this._onUp);
  }
  snapshot() {
    return {
      up:this.keys.has('KeyW')||this.keys.has('ArrowUp'),
      down:this.keys.has('KeyS')||this.keys.has('ArrowDown'),
      left:this.keys.has('KeyA')||this.keys.has('ArrowLeft'),
      right:this.keys.has('KeyD')||this.keys.has('ArrowRight'),
      sprint:this.keys.has('ShiftLeft')||this.keys.has('ShiftRight'),
      jump:this.keys.has('Space'),
      axisX:this.axisX * this.sensitivity,
      axisY:this.axisY * this.sensitivity,
    };
  }
  bindVirtualStick(root, knob) {
    if (!root || !knob) return;
    let pointer = null;
    const reset = () => { pointer=null; this.axisX=0; this.axisY=0; knob.style.transform='translate(-50%,-50%)'; };
    const move = e => {
      const r=root.getBoundingClientRect();
      const dx=e.clientX-(r.left+r.width/2), dy=e.clientY-(r.top+r.height/2);
      const max=r.width*0.32;
      const mag=Math.hypot(dx,dy)||1;
      const k=Math.min(1,max/mag);
      const x=dx*k,y=dy*k;
      this.axisX=x/max; this.axisY=y/max;
      knob.style.transform=`translate(calc(-50% + ${x}px),calc(-50% + ${y}px))`;
    };
    root.addEventListener('pointerdown', e => { pointer=e.pointerId; root.setPointerCapture?.(pointer); move(e); });
    root.addEventListener('pointermove', e => { if (e.pointerId===pointer) move(e); });
    root.addEventListener('pointerup', reset); root.addEventListener('pointercancel', reset);
  }
  dispose(){ this.target?.removeEventListener?.('keydown',this._onDown); this.target?.removeEventListener?.('keyup',this._onUp); }
}
