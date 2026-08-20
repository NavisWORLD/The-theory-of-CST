# Mars Synapse: Red Genesis Architecture

Red Genesis is the browser-playable game layer of the CST origin repository. It is deliberately separated from the legacy Python and Unity simulator so the game can run from static files while still preserving a clean bridge back into the older architecture.

## Runtime boundary

```text
Input
  ↓
Game runtime (`game/src/game.js`)
  ├─ World/environment
  ├─ Player + rover + survival
  ├─ Economy + structures + production
  ├─ CST 12-channel state + entropy + psiProxy
  ├─ Progression + research + achievements
  └─ Save-state model
        ↓
Presentation adapters
  ├─ Canvas renderer
  ├─ DOM HUD and menus
  ├─ Keyboard/touch input
  └─ Web Audio / opt-in microphone
```

The runtime owns authoritative state. Rendering and UI consume the state and emit semantic actions rather than owning simulation truth.

## Fixed-step loop

`createGameRuntime()` exposes a browser-independent runtime that can also run under Node tests. Browser animation time is accumulated and the simulation advances in fixed 1/60-second steps, keeping deterministic behavior separate from display frame rate.

## Deterministic world

`world.js` derives terrain texture, rocks, resources, anomalies and hazard timing from a seed. Resource sampling at the same seed and coordinates is reproducible.

## Mars physics

Player-scale gravity is 3.71 m/s². The game uses simplified local movement, jump arcs, rover acceleration/drag and bounded world collision. The older cosmic pairwise-force model remains part of the simulator lineage and is not applied directly to EVA or rover physics.

## CST gameplay layer

`cst.js` maps conventional game systems into twelve normalized channels. Neighboring channels influence each other through bounded smoothing. Infrastructure nodes and settlements form a weighted graph whose links strengthen through repeated local cooperation and decay slowly when unused.

Entropy is a bounded instability proxy derived from hazards, damage, reserve risk, disconnection and rapid state mismatch. `psiProxy` combines energy, coupling, memory, adaptation and instability opportunity for gameplay and presentation. It is not described as a physical measurement.

## Memory

Important events are stored in an ordered local ledger. Each entry includes the previous entry identifier. This preserves the lineage of the original memory-node chaining idea without presenting the local save mechanism as a decentralized blockchain.

## Research

Research is mechanically consequential:

- mobility research reduces rover battery consumption
- energy research raises battery storage capacity
- advanced construction is gated behind relevant technology branches
- research count contributes to knowledge and adaptation channels

## Save contract

Version 1 uses `schemaVersion: 1`. The canonical field shape is documented in `game/schema/game-state.schema.json`. Invalid or incompatible saves fail closed and allow the player to start a fresh game.

## Browser / Unity split

The browser game is the reference rules implementation. `game/UNITY_BRIDGE.md` maps its state into proposed Unity components for a future high-end 3D build while preserving the existing `CosmicEngine.cs`, `PlanetFactory.cs`, `MemoryRift.cs` and Python simulator lineage.

## Verification

```bash
cd game
npm test
npm run check
```

The Node suite exercises deterministic world generation, movement, survival, rover behavior, CST bounds and links, memory chaining, economy, technology effects, progression, victory conditions, saves and headless runtime integration.
