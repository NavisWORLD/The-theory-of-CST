import test from 'node:test';
import assert from 'node:assert/strict';
import { projectCinematic, atmosphereProfile } from '../src/visuals.js';

test('cinematic projection keeps player anchor low-left and shrinks distant objects',()=>{
  const near=projectCinematic({dx:0,dy:0,width:1600,height:900,zoom:1});
  const far=projectCinematic({dx:0,dy:-500,width:1600,height:900,zoom:1});
  assert.ok(near.x < 1600/2);
  assert.ok(near.y > 900/2);
  assert.ok(far.scale < near.scale);
  assert.ok(far.y < near.y);
});

test('dust storm increases haze and dust opacity',()=>{
  const clear=atmosphereProfile({solar:.8,dust:.1,storm:null});
  const storm=atmosphereProfile({solar:.4,dust:.85,storm:{type:'dust',severity:.8}});
  assert.ok(storm.haze > clear.haze);
  assert.ok(storm.dustOpacity > clear.dustOpacity);
  assert.ok(storm.sunStrength < clear.sunStrength);
});
