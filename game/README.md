# Mars Synapse: Red Genesis

A browser-playable survival, exploration, colony-building, and CST planetary-network game evolved from the original CST simulator in this repository.

## Play locally

From the repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/game/`.

No Python CST backend, account, API key, CDN, or external game engine is required for normal browser play.

## Core loop

**Land → scan → gather → survive → craft → build → research → establish settlements → strengthen CST/logistics links → stabilize the twelve-channel planetary state → activate Red Genesis.**

Mars gameplay includes 3.71 m/s² local gravity, rover traversal, oxygen and suit energy, radiation, temperature stress, dust and solar hazards, procedural resources/anomalies, structures, production, research, missions, achievements, saves, touch controls, synthesized audio, and optional local microphone reactivity.

## Controls

Desktop: WASD move, Shift sprint, Space jump, E gather/use, F scan, R rover, B build, I inventory, M network map, Tab CST overview, Esc pause.

Touch controls are built into the game UI.

## Verification

```bash
cd game
npm test
npm run check
```

The behavior suite covers the simulation core, CST state, economy, player/survival behavior, progression, saves, and browser-game smoke paths.

## Scientific boundary

The twelve CST channels are normalized computational/gameplay coordinates. `psiProxy` is explicitly a gameplay proxy. Neither is presented as evidence for twelve physical spacetime dimensions or a newly measured physical field.

## Architecture

- `src/game.js` owns the runtime step/action interface.
- `src/world.js` generates deterministic Mars terrain/resources and environment dynamics.
- `src/player.js` handles movement, rover and survival systems.
- `src/economy.js` handles crafting, construction, production and settlements.
- `src/cst.js` handles the 12-channel state, entropy, psi proxy, links and memory.
- `src/progression.js` handles missions, research, achievements and endgame.
- `src/save.js` handles autosave/manual save state.
- `src/render.js`, `src/ui.js`, `src/input.js`, and `src/audio.js` form the browser presentation layer.

See `UNITY_BRIDGE.md` for the 3D Unity expansion contract.
