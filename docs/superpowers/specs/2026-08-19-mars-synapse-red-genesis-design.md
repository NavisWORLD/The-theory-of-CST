# Mars Synapse: Red Genesis

## Design specification

Date: 2026-08-19
Repository: `NavisWORLD/The-theory-of-CST`
Branch: `mars-synapse-red-genesis`

## 1. Product statement

**Mars Synapse: Red Genesis** is a playable survival, exploration, building, and planetary-systems game evolved directly from the original CST Earth/cosmic simulation lineage in this repository.

The original project generated worlds and entities from CST-inspired state, entropy, audio, procedural terrain, gravitational interactions, memory, and synaptic-style values. Red Genesis keeps those ideas but changes the player contract: the player is no longer only watching a simulation. They land on Mars, move through it, gather resources, survive hazards, build a colony, make strategic choices, unlock technology, connect settlements, and ultimately stabilize a planetary CST network.

The game must feel complete as a game even to a player who knows nothing about CST. CST is the systems layer that makes the world behave differently, not a requirement for understanding the controls.

## 2. Core fantasy

You are the first operator of a self-growing Martian settlement network. Every machine, habitat, rover, relay, storm, resource field, memory event, and colony decision becomes part of a twelve-channel planetary state.

The fantasy is not merely “terraform Mars.” The fantasy is:

> Build a civilization whose infrastructure begins behaving like a connected planetary nervous system.

The player starts with a damaged landing package and one habitat. By the end of a successful campaign, three or more settlements form a stable linked network whose stored history, energy, logistics, environmental control, and CST coupling are strong enough to activate the Red Genesis planetary synapse.

## 3. Design pillars

### 3.1 Play first

The game must be understandable from movement, goals, feedback, and consequences. No theory reading is required to enjoy it.

### 3.2 CST must cause gameplay

CST values cannot exist only as decorative HUD numbers. They must alter actual systems: efficiency, hazard intensity, research discovery, structure coupling, resource behavior, colony morale/stability proxies, and endgame conditions.

### 3.3 Mars must feel physically distinct

Mars uses lower gravity, long ballistic motion, thin-atmosphere hazards, thermal swings, dust accumulation, radiation exposure, scarce water, solar variability, and power/oxygen logistics. These systems are simplified for playability but should preserve the character of Mars.

### 3.4 Every run remembers

The simulation accumulates persistent event history. Important actions leave memory entries that influence later CST state and provide a visible colony chronicle.

### 3.5 The original simulator remains lineage, not dead code

The legacy Python and Unity files remain preserved. New game systems should map cleanly onto concepts already present in `cst_engine.py`, `CosmicEngine.cs`, `PlanetFactory.cs`, `MemoryRift.cs`, `NPCBehavior.cs`, and the 12D CST documentation.

## 4. Recommended implementation strategy

Use a **hybrid architecture**.

### 4.1 Browser game is the primary playable build

The first complete game ships as a self-contained browser experience under a new `game/` directory. It should run from a static server and require no backend for normal play.

Recommended technology:

- HTML5 Canvas for rendering
- vanilla JavaScript modules
- Web Audio for game audio and optional microphone-reactive CST input
- localStorage or IndexedDB for saves
- deterministic seeded world generation
- no mandatory external CDN or framework

The browser build is the public reference implementation because anyone can open it immediately.

### 4.2 Unity compatibility is a parallel architecture target

The browser systems are designed around clear domain objects and data schemas so they can later be mirrored in Unity without rewriting the game design.

The existing Unity scripts remain the high-end 3D lineage. New Unity-specific implementation should eventually consume the same conceptual state:

- player survival state
- colony resource state
- structure network
- Mars environment
- CST 12D state
- memory events
- progression flags

### 4.3 Python CST backend remains optional

The legacy Python simulation is not required for the browser build. A future bridge may let the game import/export CST state to the Python engine. The game must never fail because Python, PyAudio, SciPy, or the local socket server is unavailable.

## 5. Game loop

The core minute-to-minute loop is:

1. Explore terrain.
2. Locate water ice, regolith metals, silica, carbon-bearing material, and salvage.
3. Manage oxygen, suit energy, health, thermal load, and radiation exposure.
4. Return resources to the base or rover.
5. Craft tools and place infrastructure.
6. Generate and store power.
7. Expand oxygen, water, storage, and communications capacity.
8. Complete missions and research objectives.
9. Survive environmental events.
10. Link structures into local synaptic clusters.
11. Establish remote settlements.
12. Connect settlements into a planetary network.
13. Stabilize the twelve-channel CST state.
14. Activate Red Genesis.

The player should always have a short-term task, a medium-term expansion goal, and a visible long-term campaign objective.

## 6. Player systems

### 6.1 Movement

The player can move on foot and operate a rover.

On foot:

- WASD or virtual thumbstick movement
- mouse/touch aim
- sprint with oxygen/energy cost
- jump with Mars-style lower gravity
- inertia retained lightly enough to feel different without becoming frustrating
- collision with terrain, rocks, structures, and hazards

Rover:

- higher speed
- battery consumption
- cargo capacity
- traction loss during heavy dust conditions
- collision damage
- upgradeable motor, battery, storage, shielding, scanner, and suspension

### 6.2 Survival meters

The player tracks:

- health
- oxygen
- suit battery
- temperature stress
- radiation dose

The game should avoid constant meter babysitting. Systems become dangerous when the player ignores planning, not every thirty seconds.

### 6.3 Inventory

Inventory uses stackable resource slots plus equipped tools.

Primary resources:

- water ice
- regolith
- iron/nickel feedstock
- silica
- carbon compounds
- electronics salvage
- rare anomaly samples

Processed materials:

- water
- oxygen
- metal plates
- glass/composites
- wiring/electronics
- fuel/reactants
- CST lattice components

## 7. Mars environment and physics

### 7.1 Gravity

The gameplay gravity target is approximately 0.38 Earth gravity. It affects jump arcs, falling, rover suspension, loose debris, and ballistic tools.

### 7.2 Day and night

The simulation uses a Martian sol cycle scaled to a playable duration. Light and solar generation vary over the cycle.

Night introduces:

- lower temperature
- weaker solar power
- greater battery dependence
- altered visibility
- increased pressure on life support planning

### 7.3 Dust

Dust is a persistent gameplay system rather than a cosmetic overlay.

Dust can:

- reduce solar output
- lower visibility
- reduce rover traction
- accumulate on exposed equipment
- increase maintenance demand
- change CST entropy through environmental instability

Dust storms are forecastable enough for player planning but retain uncertainty.

### 7.4 Radiation

Radiation accumulates while exposed. Habitats, terrain cover, upgraded suits, and shielding reduce dose.

### 7.5 Temperature

Temperature stress is based on time of sol, storms, shelter, suit integrity, and location. The simulation is deliberately game-scaled rather than a laboratory thermal model.

### 7.6 Resource geology

Resources are generated from seeded regional fields rather than pure random pickups. Scanners reveal confidence zones. This creates meaningful exploration and makes settlement placement strategic.

## 8. Colony building

Buildings snap to terrain-valid positions and connect through power/logistics range.

Initial structure set:

- habitat
- solar array
- battery bank
- oxygen extractor
- water processor
- storage container
- fabrication bench
- communications relay
- rover bay
- research station
- radiator/thermal controller
- radiation shelter

Advanced structures:

- greenhouse
- wind/dust harvester experimental unit
- nuclear auxiliary power module
- deep ice drill
- automated miner
- drone pad
- orbital uplink
- CST relay tower
- memory archive
- synaptic lattice hub
- settlement core

Placement must matter because structures exchange power, logistics, and CST coupling over distance.

## 9. Economy and production

Production uses simple recipes with visible inputs and outputs.

Each structure has:

- construction cost
- power draw/generation
- storage capacity where relevant
- maintenance state
- CST signature
- efficiency modifier

Efficiency is affected by conventional factors first, such as power availability, maintenance, storm conditions, and logistics. CST can modify the result but never replace understandable simulation rules.

## 10. CST 12-channel planetary state

The game defines a normalized twelve-channel state vector `S[0..11]` in the range 0 to 1.

The channels are gameplay coordinates, not claims of twelve physical spacetime dimensions.

### D01 Energy

Power generation, reserves, and grid resilience.

### D02 Matter

Material stockpiles, extraction, fabrication throughput.

### D03 Water

Ice access, liquid water reserve, recycling reliability.

### D04 Atmosphere

Oxygen production, sealed volume, pressure/life-support stability.

### D05 Thermal

Temperature control capacity and thermal stress.

### D06 Radiation

Shielding resilience and accumulated exposure risk.

### D07 Mobility

Rover health, route access, transportation capacity.

### D08 Knowledge

Research progress, scanned regions, discovered anomalies.

### D09 Memory

Persistence and quality of colony historical state, archives, recovered records.

### D10 Social/Settlement Coherence

A non-sentience gameplay proxy derived from habitat redundancy, mission completion, communication links, and settlement health. It is not a psychological or consciousness claim.

### D11 Network Coupling

Number, quality, and distance efficiency of links between structures and settlements.

### D12 Adaptation

How effectively the colony has learned from hazards and repeated conditions through upgrades, automation, and remembered events.

## 11. CST dynamics

The state update follows the conceptual form already documented in CST:

```text
state(t+1) = F(state(t), input(t), memory(t), neighbors(t), coupling(t), dynamics(t))
```

For the game, each channel updates from:

- current infrastructure
- current resource reserves
- environmental conditions
- recent player actions
- neighboring channel influence
- stored memory events
- settlement network topology
- bounded decay/recovery

All channels are clamped to `[0,1]`.

### 11.1 Synaptic links

Structures and settlements become graph nodes. Edges gain synaptic weight from:

- physical/logistics connection
- repeated resource exchange
- communication uptime
- mission interdependence
- shared hazard survival

Frequently cooperating nodes strengthen their link. Unused or broken links decay slowly.

### 11.2 Entropy

Entropy is a bounded world-instability proxy driven by:

- active storms
- damaged infrastructure
- low reserves
- disconnected settlements
- rapid state change
- unresolved hazards

Higher entropy can increase anomaly frequency, reduce network efficiency, and create rare research opportunities. It is not simply a punishment meter.

### 11.3 Psi proxy

The game uses a clearly labeled **CST psi gameplay proxy**. It combines normalized energy availability, network coupling, memory depth, instability, and environmental load.

It must never be described in-game as a physically measured new field.

Psi affects:

- strength of CST relay effects
- discovery probability for anomaly events
- visual/audio world response
- endgame readiness

### 11.4 Memory

Major events are recorded in an append-only game memory ledger:

- first water discovery
- first habitat completion
- storms survived
- critical failures
- settlement launches
- research breakthroughs
- player rescue/recovery events
- major network activation events

Each memory entry includes a deterministic identifier and the previous entry identifier, echoing the legacy hash-chained memory-node design while remaining a local game save feature rather than a blockchain claim.

## 12. Hazards

Environmental hazards:

- dust storms
- solar/radiation events
- cold snaps
- equipment failure
- rover breakdown
- oxygen leaks
- power-grid collapse
- terrain hazards

Procedural anomaly hazards:

- unstable CST resonance zones
- buried electromagnetic/mineral anomalies
- memory echoes represented as environmental events

No supernatural explanation is required. Anomalies are framed as game-world research phenomena.

## 13. Missions and narrative

The campaign uses mission chains rather than a long passive tutorial.

### Act I: Touchdown

- restore lander power
- repair oxygen system
- locate first ice deposit
- establish safe night reserve

### Act II: First Pulse

- build research station
- deploy first CST relay
- survive first major storm
- recover a damaged expedition cache

### Act III: The Distance Between Nodes

- construct rover upgrades
- establish second settlement
- create reliable resource transfer
- maintain two linked settlements through a crisis

### Act IV: Red Network

- establish third settlement
- build memory archives
- connect all settlements through relays
- raise network coupling and adaptation

### Act V: Genesis

- stabilize all twelve channels above campaign thresholds
- survive a final large planetary storm sequence
- maintain the network long enough to activate Red Genesis

The ending shows the planet as a connected human-built network. The victory claim is that the player successfully stabilized the game system, not that CST has been physically proven.

## 14. Progression and technology

Research points come from:

- scanning terrain
- anomaly samples
- surviving new hazard classes
- building new structure categories
- maintaining high-performing systems
- mission completion

Tech branches:

- Survival
- Mobility
- Extraction
- Energy
- Habitat
- Automation
- Science
- CST Network

Meaningful unlock examples:

- better suit oxygen recycling
- shielded rover cabin
- larger batteries
- deep ice drilling
- automated repair drones
- compact reactor
- settlement cargo automation
- long-range relay
- adaptive storm routing
- synaptic lattice hub

## 15. Failure and recovery

The player can fail locally without losing the campaign immediately.

If the player collapses away from base:

- a recovery cost is applied
- time advances
- some carried materials may be lost
- the event enters memory

Campaign failure occurs only if the colony enters unrecoverable total life-support collapse with no functioning settlement core or stored recovery path.

A failed campaign can be restarted from the same seed or a new seed.

## 16. Save system

Support three manual save slots plus autosave.

Save data includes:

- schema version
- world seed
- player state
- rover state
- inventory
- structures
- settlement graph
- resources
- mission state
- research unlocks
- environment state
- CST 12D vector
- entropy
- psi proxy
- synaptic weights
- memory ledger
- achievements
- playtime

Autosave triggers:

- after completing a mission
- after constructing a settlement core
- after major hazard resolution
- every configurable interval when safe

Corrupt or incompatible save data must fail gracefully and offer a fresh game rather than locking the page.

## 17. Controls

Desktop:

- WASD movement
- mouse aim/interact
- Shift sprint
- Space jump
- E interact
- F scanner
- I inventory
- B build mode
- M map
- Tab mission/CST overview
- Esc pause

Mobile:

- left virtual stick
- right drag look/aim
- context interact button
- jump button
- scanner button
- expandable action/build bar

UI must remain usable on iPhone-sized screens.

## 18. User interface

Primary HUD:

- health
- oxygen
- energy
- compact radiation/temperature warnings
- current objective
- selected tool
- contextual interaction prompt

Secondary panels:

- inventory/crafting
- build menu
- Mars map
- research tree
- settlement network
- CST 12D field
- memory chronicle
- mission log
- settings

The CST panel should visualize twelve channels as a responsive radial or ring system, with graph links between settlements and clear explanations of which gameplay systems are influencing each channel.

## 19. Audio and visual language

Visual identity:

- dark basalt and iron-red terrain
- pale dusty sky
- luminous cyan/amber CST network elements
- restrained glass HUD
- readable hazard colors
- strong silhouettes for structures

Audio:

- wind/dust ambience
- suit breathing under stress
- rover motor and suspension
- structure hums
- relay pulses
- storm warning cues
- music intensity tied subtly to entropy and network coupling

Optional microphone mode may modulate cosmetic or non-critical CST signal behavior only after explicit player activation. It must not be required for progression.

## 20. Accessibility and settings

Required settings:

- master/music/effects volume
- reduced motion
- screen shake toggle
- high-contrast HUD option
- text size
- mobile control sensitivity
- keyboard remapping where practical
- autosave frequency
- performance quality mode

The player must be able to pause fully in single-player mode.

## 21. Achievements

Examples:

- First Breath: restore oxygen after landing
- Red Water: process first usable water
- Long Way Home: drive a major expedition distance and return
- Storm Memory: survive a severe dust storm
- Two Minds: link two settlements
- Twelve Alive: bring all twelve CST channels above minimum stability simultaneously
- Red Genesis: complete the campaign
- No One Left Behind: finish a campaign without total settlement-core loss

## 22. Technical module boundaries for the browser build

Proposed directory:

```text
game/
  index.html
  styles.css
  src/
    main.js
    core/
      Game.js
      Loop.js
      Input.js
      SaveSystem.js
      SeededRandom.js
    world/
      MarsWorld.js
      Terrain.js
      Environment.js
      ResourceField.js
      HazardDirector.js
    player/
      Player.js
      Rover.js
      Inventory.js
      Survival.js
    colony/
      BuildSystem.js
      Structure.js
      Settlement.js
      Production.js
      PowerGrid.js
    cst/
      CSTState.js
      SynapticGraph.js
      MemoryLedger.js
      CSTMetrics.js
    progression/
      Missions.js
      Research.js
      Achievements.js
    ui/
      HUD.js
      Panels.js
      MobileControls.js
    audio/
      AudioSystem.js
  tests/
    cst-state.test.js
    save-system.test.js
    production.test.js
    missions.test.js
```

Each module owns one clear responsibility. Rendering must not contain simulation logic that cannot be tested independently.

## 23. Data flow

Per fixed simulation tick:

1. Read player input.
2. Update player/rover physics.
3. Update survival state.
4. Update Mars environment.
5. Update hazards.
6. Update structures and production.
7. Update settlement graph.
8. Append any major memory events.
9. Update CST 12D state.
10. Update entropy and psi proxy.
11. Evaluate missions/research/achievements.
12. Render interpolated visual state.
13. Autosave when eligible.

A fixed-step simulation is preferred so gameplay is deterministic enough for reproducible seeded testing across frame rates.

## 24. Error handling

- Missing Web Audio support falls back to silent/limited audio mode.
- Denied microphone permission never blocks the game.
- localStorage/IndexedDB failure disables saving with a visible warning but permits a temporary session.
- Invalid save versions are rejected safely.
- NaN/Infinity values in physics or CST state are clamped/recovered and logged to the in-game debug console in development mode.
- Missing optional assets use generated/fallback visuals.

## 25. Testing strategy

Unit tests must cover:

- deterministic seeded generation
- CST channel bounds
- synaptic weight strengthening/decay
- entropy bounds
- psi proxy finite values
- production accounting
- power-grid behavior
- survival depletion/recovery
- save/load round trip
- save migration/version rejection
- mission progression
- victory/failure conditions

Simulation tests should run hundreds of accelerated ticks to verify:

- no negative impossible resource stores
- no NaN/Infinity propagation
- structures cannot create free material
- hazards resolve
- colonies can reach victory with a valid strategy
- unrecoverable failure is actually possible under severe neglect

Browser smoke testing must verify desktop and touch input paths.

## 26. Performance targets

Baseline target devices include modern mobile browsers and ordinary laptops.

Design targets:

- 60 FPS preferred desktop
- 30 FPS minimum supported mobile under reduced quality
- fixed simulation tick decoupled from render rate
- terrain/resource chunks streamed around the player
- offscreen structures simulated at reduced detail
- particle counts scaled by quality setting
- no requirement for WebGL-heavy external engines in the first browser build

## 27. Scope guardrails

The first complete campaign does **not** require:

- multiplayer
- full orbital mechanics
- photorealistic Mars
- realistic chemistry for every resource
- an LLM NPC system
- cryptocurrency/blockchain
- mandatory online services
- a full 3D Unity remake before the browser game is playable
- claims that CST has been experimentally validated as new physics

These exclusions prevent the project from becoming impossible to finish.

## 28. Definition of done

Red Genesis is considered a complete first game when a fresh player can:

1. Open the game from a static server.
2. Start a new seeded campaign.
3. Complete an onboarding landing sequence.
4. Walk, jump, scan, gather, and use inventory.
5. Drive a rover.
6. Manage oxygen, health, power, radiation, and temperature.
7. Build and operate a functioning base.
8. Experience day/night and at least two hazard classes.
9. Craft and research upgrades.
10. Establish at least three settlements.
11. See CST channels respond to actual game state.
12. Observe synaptic links strengthen/decay from settlement interaction.
13. Read persistent memory events.
14. Save, reload, and continue correctly.
15. Lose through genuine colony collapse.
16. Win by activating Red Genesis after satisfying all endgame conditions.
17. Play on desktop and mobile controls.
18. Complete automated simulation/unit tests without numerical instability.

## 29. Final design decision

The project will move forward as a **CST-native Mars survival and colony game**, not as a visualization demo.

The browser build is the first complete playable product. The existing Python and Unity simulator remain the lineage and future bridge. Game mechanics remain understandable without CST, while CST creates a second-order systems layer that makes player history, infrastructure topology, memory, and adaptation matter to the world.

The campaign endpoint is not “prove the theory.” It is **build the Red Genesis network and make Mars behave like one connected engineered system**.
