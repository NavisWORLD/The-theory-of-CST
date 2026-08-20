# Mars Synapse: Red Genesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete browser-playable Mars survival, exploration, building, progression, and CST-driven colony game that evolves the original CST simulator into a real game while preserving the legacy Python and Unity lineage.

**Architecture:** The playable build lives in `game/` and uses small vanilla JavaScript modules with deterministic state updates and a Canvas renderer. Simulation logic is pure and testable in Node; rendering, input, audio, and persistence are adapters around that core. The original Python and Unity files remain untouched except for documentation links and optional compatibility notes.

**Tech Stack:** HTML5 Canvas, ES modules, Web Audio, localStorage, Node.js built-in test runner, vanilla CSS, existing Python/Unity CST lineage.

**Spec:** `docs/superpowers/specs/2026-08-19-mars-synapse-red-genesis-design.md`

## Global Constraints

- The game must be enjoyable without knowing CST.
- CST state must alter gameplay rather than exist only as decorative telemetry.
- The browser build must run without Python, PyAudio, SciPy, a socket server, or an external CDN.
- The twelve CST channels are normalized computational gameplay coordinates, not claims of twelve physical spacetime dimensions.
- The CST psi value used by the game must be labeled a gameplay proxy, not a measured physical field.
- Mars uses approximately 0.38 Earth gravity in the movement model.
- Save data must include a schema version and fail gracefully when invalid.
- Desktop and touch controls must both be supported.
- Optional microphone input must require explicit activation and must never be required for progression.
- Existing legacy CST Python and Unity files remain preserved.

---

## File map

- `game/index.html` - game shell and accessible UI containers.
- `game/styles.css` - HUD, panels, title screen, mobile controls, responsive layout.
- `game/src/constants.js` - tunable game constants and recipe/building definitions.
- `game/src/random.js` - deterministic seeded PRNG and hash helpers.
- `game/src/state.js` - canonical new-game state, validation, cloning, serialization shape.
- `game/src/world.js` - deterministic Mars terrain/resource fields, hazards, day/sol state.
- `game/src/player.js` - astronaut movement, rover movement, survival meter updates, collisions.
- `game/src/economy.js` - gathering, inventory, recipes, building placement, production.
- `game/src/cst.js` - twelve-channel state, synaptic graph, entropy, psi gameplay proxy, memory ledger.
- `game/src/progression.js` - missions, research unlocks, achievements, campaign victory/failure.
- `game/src/save.js` - localStorage slots, autosave, migration/validation guards.
- `game/src/audio.js` - Web Audio ambience/events and opt-in microphone-reactive cosmetic signal.
- `game/src/input.js` - keyboard, pointer, touch-stick, action state.
- `game/src/render.js` - Canvas world, player, structures, particles, HUD-world overlays.
- `game/src/ui.js` - DOM HUD/panels/tutorial/pause/title/settings orchestration.
- `game/src/game.js` - fixed-step loop and module integration.
- `game/tests/*.test.mjs` - Node behavior tests.
- `game/README.md` - controls, local run instructions, CST gameplay mapping.
- `.github/workflows/ci.yml` - add Node game tests beside existing Python verification.

---

### Task 1: Deterministic core state and world generation

**Files:**
- Create: `game/src/random.js`
- Create: `game/src/constants.js`
- Create: `game/src/state.js`
- Create: `game/src/world.js`
- Test: `game/tests/core.test.mjs`

**Interfaces:**
- Produces: `createRng(seed) -> () => number`, `hashString(text) -> uint32`, `createInitialState(seed) -> GameState`, `generateWorld(seed) -> WorldState`, `sampleResource(world,x,y) -> ResourceSample`, `updateEnvironment(state,dt) -> void`.

- [ ] **Step 1: Write failing deterministic-world tests**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { generateWorld, sampleResource } from '../src/world.js';

test('same seed creates same initial state', () => {
  assert.deepEqual(createInitialState('ARES-01'), createInitialState('ARES-01'));
});

test('same world coordinate returns same resource sample', () => {
  const a = generateWorld('ARES-01');
  const b = generateWorld('ARES-01');
  assert.deepEqual(sampleResource(a, 120, -45), sampleResource(b, 120, -45));
});
```

- [ ] **Step 2: Run RED**

Run: `node --test game/tests/core.test.mjs`
Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement deterministic PRNG, constants, initial state, seeded resource fields, and sol/environment state**

Required initial state fields:

```js
{
  schemaVersion: 1,
  seed: 'ARES-01',
  time: { elapsed: 0, sol: 1, solPhase: 0.24 },
  player: { x: 0, y: 0, vx: 0, vy: 0, health: 100, oxygen: 100, energy: 100, radiation: 0, temperatureStress: 0, mode: 'foot' },
  inventory: { ice: 0, regolith: 0, metal: 0, silica: 0, carbon: 0, salvage: 0, anomaly: 0 },
  structures: [],
  settlements: [],
  research: { points: 0, unlocked: ['survival-1'] },
  missions: { act: 1, completed: [] },
  achievements: [],
  cst: { channels: Array(12).fill(0.1), entropy: 0.2, psiProxy: 0, links: [], memory: [] },
  environment: { dust: 0.15, radiation: 0.18, temperature: -35, storm: null },
  flags: { paused: false, gameOver: false, victory: false }
}
```

- [ ] **Step 4: Run GREEN**

Run: `node --test game/tests/core.test.mjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game/src/random.js game/src/constants.js game/src/state.js game/src/world.js game/tests/core.test.mjs
git commit -m "feat: add deterministic Mars world core"
```

---

### Task 2: Player physics, rover, survival, gathering

**Files:**
- Create: `game/src/player.js`
- Modify: `game/src/world.js`
- Test: `game/tests/player.test.mjs`

**Interfaces:**
- Consumes: `GameState`, `sampleResource`.
- Produces: `stepPlayer(state,input,dt)`, `stepSurvival(state,dt)`, `gatherAtPlayer(state,world)`, `toggleVehicle(state)`.

- [ ] **Step 1: Write failing Mars-gravity and gathering tests**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { stepPlayer, gatherAtPlayer } from '../src/player.js';
import { generateWorld } from '../src/world.js';

test('jump uses Mars gravity and returns to ground', () => {
  const s = createInitialState('JUMP');
  stepPlayer(s, { jump: true }, 1/60);
  assert.ok(s.player.verticalVelocity > 0);
  for (let i=0;i<600;i++) stepPlayer(s, {}, 1/60);
  assert.equal(s.player.altitude, 0);
});

test('gathering adds a deterministic resource stack', () => {
  const s = createInitialState('MINE');
  const w = generateWorld('MINE');
  const before = Object.values(s.inventory).reduce((a,b)=>a+b,0);
  gatherAtPlayer(s, w);
  const after = Object.values(s.inventory).reduce((a,b)=>a+b,0);
  assert.ok(after > before);
});
```

- [ ] **Step 2: Run RED**

Run: `node --test game/tests/player.test.mjs`
Expected: FAIL because player module is absent.

- [ ] **Step 3: Implement foot/rover acceleration, drag, approximately 3.71 m/s² vertical gravity, jump, sprint cost, rover battery/cargo differences, oxygen/energy/radiation/temperature updates, collapse/recovery, and gathering cooldown**

- [ ] **Step 4: Run GREEN and all core tests**

Run: `node --test game/tests/*.test.mjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game/src/player.js game/src/world.js game/tests/player.test.mjs
git commit -m "feat: add Mars movement survival and gathering"
```

---

### Task 3: CST twelve-channel state, synaptic graph, entropy, psi proxy, memory

**Files:**
- Create: `game/src/cst.js`
- Test: `game/tests/cst.test.mjs`

**Interfaces:**
- Produces: `updateCst(state,dt)`, `recordMemory(state,type,message,data={})`, `strengthenLink(state,a,b,amount)`, `computeCstChannels(state) -> number[12]`.

- [ ] **Step 1: Write failing bounded-state and link-learning tests**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { updateCst, recordMemory, strengthenLink } from '../src/cst.js';

test('CST channels remain bounded', () => {
  const s = createInitialState('CST');
  s.inventory.ice = 99999;
  s.environment.dust = 1;
  updateCst(s, 1);
  assert.equal(s.cst.channels.length, 12);
  assert.ok(s.cst.channels.every(v => v >= 0 && v <= 1));
});

test('repeated cooperation strengthens one graph link', () => {
  const s = createInitialState('LINK');
  strengthenLink(s, 'hab-1', 'solar-1', 0.1);
  strengthenLink(s, 'hab-1', 'solar-1', 0.1);
  assert.equal(s.cst.links.length, 1);
  assert.ok(s.cst.links[0].weight > 0.1);
});

test('memory entries form a previous-id chain', () => {
  const s = createInitialState('MEM');
  recordMemory(s, 'first-water', 'Water located');
  recordMemory(s, 'storm', 'Storm survived');
  assert.equal(s.cst.memory[1].previousId, s.cst.memory[0].id);
});
```

- [ ] **Step 2: Run RED**

Run: `node --test game/tests/cst.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement D01-D12 mapping, bounded smoothing, link decay/strengthening, entropy from hazards/damage/disconnection, psi gameplay proxy, deterministic hash-linked memory entries**

Use the documented conceptual update:

```text
state(t+1) = F(state(t), input(t), memory(t), neighbors(t), coupling(t), dynamics(t))
```

- [ ] **Step 4: Run GREEN**

Run: `node --test game/tests/*.test.mjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game/src/cst.js game/tests/cst.test.mjs
git commit -m "feat: make CST state drive Mars gameplay"
```

---

### Task 4: Economy, crafting, building, production, settlement graph

**Files:**
- Create: `game/src/economy.js`
- Modify: `game/src/constants.js`
- Test: `game/tests/economy.test.mjs`

**Interfaces:**
- Produces: `canAfford`, `craft`, `placeStructure`, `stepProduction`, `createSettlement`, `getNetworkEfficiency`.

- [ ] **Step 1: Write failing economy tests**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { placeStructure, stepProduction } from '../src/economy.js';

test('placing a structure spends materials and creates a node', () => {
  const s = createInitialState('BUILD');
  Object.assign(s.inventory, { regolith: 50, metal: 50, silica: 50, salvage: 50, ice: 50, carbon: 50 });
  const ok = placeStructure(s, 'solar', 10, 10);
  assert.equal(ok, true);
  assert.equal(s.structures.length, 1);
});

test('powered oxygen extractor raises oxygen reserve', () => {
  const s = createInitialState('PROD');
  s.structures.push({ id:'solar-1', type:'solar', x:0, y:0, health:1 });
  s.structures.push({ id:'oxygen-1', type:'oxygen', x:5, y:0, health:1 });
  s.inventory.ice = 20;
  const before = s.resources.oxygen;
  stepProduction(s, 10);
  assert.ok(s.resources.oxygen > before);
});
```

- [ ] **Step 2: Run RED**

Run: `node --test game/tests/economy.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement recipes, structure catalog, placement validation, power budget, water/oxygen/material processing, maintenance, settlement cores, distance-based logistics, and graph coupling hooks**

Required early structures: `habitat`, `solar`, `battery`, `oxygen`, `water`, `storage`, `fabricator`, `relay`, `rover-bay`, `research`, `radiator`, `shelter`.

Required advanced structures: `greenhouse`, `reactor`, `deep-drill`, `miner`, `drone-pad`, `uplink`, `cst-relay`, `memory-archive`, `lattice-hub`, `settlement-core`.

- [ ] **Step 4: Run GREEN**

Run: `node --test game/tests/*.test.mjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game/src/economy.js game/src/constants.js game/tests/economy.test.mjs
git commit -m "feat: add colony economy and construction"
```

---

### Task 5: Missions, research, achievements, victory and failure

**Files:**
- Create: `game/src/progression.js`
- Test: `game/tests/progression.test.mjs`

**Interfaces:**
- Produces: `updateProgression(state)`, `completeMission(state,id)`, `unlockResearch(state,id)`, `evaluateCampaign(state)`.

- [ ] **Step 1: Write failing campaign tests**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { evaluateCampaign } from '../src/progression.js';

test('three linked stable settlements can trigger victory', () => {
  const s = createInitialState('WIN');
  s.settlements = [{id:'a'},{id:'b'},{id:'c'}];
  s.cst.channels = Array(12).fill(0.86);
  s.cst.links = [{a:'a',b:'b',weight:0.9},{a:'b',b:'c',weight:0.9},{a:'c',b:'a',weight:0.9}];
  s.cst.entropy = 0.2;
  evaluateCampaign(s);
  assert.equal(s.flags.victory, true);
});
```

- [ ] **Step 2: Run RED**

Run: `node --test game/tests/progression.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement five campaign acts, research branches, achievement checks, local collapse recovery, total-colony failure, and Red Genesis endgame thresholds**

- [ ] **Step 4: Run GREEN**

Run: `node --test game/tests/*.test.mjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game/src/progression.js game/tests/progression.test.mjs
git commit -m "feat: add Red Genesis campaign progression"
```

---

### Task 6: Versioned save slots and autosave

**Files:**
- Create: `game/src/save.js`
- Test: `game/tests/save.test.mjs`

**Interfaces:**
- Produces: `serializeGame(state)`, `deserializeGame(text)`, `saveSlot(storage,state,slot)`, `loadSlot(storage,slot)`, `deleteSlot(storage,slot)`.

- [ ] **Step 1: Write failing round-trip and corrupt-save tests**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState } from '../src/state.js';
import { serializeGame, deserializeGame } from '../src/save.js';

test('save round trip preserves game state', () => {
  const s = createInitialState('SAVE');
  s.inventory.ice = 42;
  const restored = deserializeGame(serializeGame(s));
  assert.equal(restored.inventory.ice, 42);
  assert.equal(restored.seed, 'SAVE');
});

test('corrupt save returns null instead of throwing', () => {
  assert.equal(deserializeGame('{not-json'), null);
});
```

- [ ] **Step 2: Run RED**

Run: `node --test game/tests/save.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement schema validation, three manual slots, autosave key, metadata summaries, and safe invalid-save behavior**

- [ ] **Step 4: Run GREEN**

Run: `node --test game/tests/*.test.mjs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game/src/save.js game/tests/save.test.mjs
git commit -m "feat: add resilient Mars game saves"
```

---

### Task 7: Playable Canvas client, HUD, touch controls, audio, tutorial

**Files:**
- Create: `game/index.html`
- Create: `game/styles.css`
- Create: `game/src/input.js`
- Create: `game/src/render.js`
- Create: `game/src/ui.js`
- Create: `game/src/audio.js`
- Create: `game/src/game.js`
- Test: `game/tests/smoke.test.mjs`

**Interfaces:**
- `game.js` owns the fixed-step loop and calls all pure simulation modules.
- Renderer consumes immutable-ish snapshots and never owns authoritative simulation state.
- UI emits semantic actions (`build`, `gather`, `scan`, `toggle-rover`, `pause`, `save`) rather than modifying simulation state directly.

- [ ] **Step 1: Write failing module smoke test**

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createGameRuntime } from '../src/game.js';

test('runtime can advance a new game without browser globals', () => {
  const runtime = createGameRuntime({ seed: 'SMOKE', headless: true });
  runtime.step(1/60, {});
  assert.equal(runtime.state.time.sol, 1);
  assert.equal(runtime.state.cst.channels.length, 12);
});
```

- [ ] **Step 2: Run RED**

Run: `node --test game/tests/smoke.test.mjs`
Expected: FAIL.

- [ ] **Step 3: Implement fixed-step runtime integration first until smoke test passes**

- [ ] **Step 4: Build Canvas renderer and complete screen flow**

Required screens/panels:

```text
Title -> New Game / Continue / Save Slots
Game HUD -> objective + survival meters + selected tool
Inventory/Crafting
Build Menu
Research
Settlement Network
CST 12D panel
Memory Chronicle
Pause/Settings
Victory / Failure
```

- [ ] **Step 5: Add desktop keyboard/pointer controls and mobile virtual stick/action buttons**

- [ ] **Step 6: Add Web Audio ambience/events and explicit opt-in microphone cosmetic reactivity**

- [ ] **Step 7: Run complete tests and syntax checks**

Run:

```bash
node --test game/tests/*.test.mjs
node --check game/src/game.js
node --check game/src/render.js
node --check game/src/ui.js
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add game/index.html game/styles.css game/src/input.js game/src/render.js game/src/ui.js game/src/audio.js game/src/game.js game/tests/smoke.test.mjs
git commit -m "feat: ship playable Mars Synapse browser client"
```

---

### Task 8: Documentation, CI, and lineage integration

**Files:**
- Create: `game/README.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Create: `docs/MARS_SYNAPSE_GAME_ARCHITECTURE.md`

**Interfaces:**
- README links must use relative paths that work on GitHub.
- CI must retain Python 3.10/3.12 verification and add a Node game test job.

- [ ] **Step 1: Update CI with a separate Node 22 job**

Required command:

```yaml
- name: Run Mars Synapse game tests
  run: node --test game/tests/*.test.mjs
```

- [ ] **Step 2: Document local play**

Required commands:

```bash
python -m http.server 8000
# open http://localhost:8000/game/
```

- [ ] **Step 3: Document CST mapping and scientific boundary**

Must explicitly state that 12D channels and psi are game/simulation variables and do not establish new physical dimensions or a measured field.

- [ ] **Step 4: Run all verification**

```bash
node --test game/tests/*.test.mjs
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add game/README.md README.md .github/workflows/ci.yml docs/MARS_SYNAPSE_GAME_ARCHITECTURE.md
git commit -m "docs: integrate Mars Synapse with CST flagship"
```

---

## Self-review result

- Spec coverage: core loop, Mars physics, survival, rover, inventory, resources, construction, production, CST state, memory, hazards, missions, research, save/load, desktop/mobile controls, audio, accessibility-facing settings, victory/failure, documentation, and CI all map to tasks above.
- Deliberately deferred scope remains deferred: multiplayer, orbital mechanics, photorealism, blockchain, mandatory Python runtime, and mandatory microphone input.
- No implementation task modifies legacy CST Python or Unity behavior.
- Public claims remain bounded: CST game values are computational mechanics and proxies, not new experimental physical evidence.
