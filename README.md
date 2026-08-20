<div align="center">

<img src="assets/cst-hero.svg" alt="Cosmic Synapse Theory 12D: state, memory, coupling, signal, and evidence" width="100%" />

# COSMIC SYNAPSE THEORY // CST

### Research program, executable simulator, and now a playable Mars world

**State. Memory. Coupling. Signal. Evidence. Play.**

[![Research Status](https://img.shields.io/badge/status-research%20prototype-7c3aed)](#scientific-boundary)
[![Mars Synapse](https://img.shields.io/badge/game-Mars%20Synapse%3A%20Red%20Genesis-e66f43)](game/)
[![State Space](https://img.shields.io/badge/CST-12D-00d4ff)](CST_Formula_Explanation.markdown)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](#run-the-original-simulator)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17574447-1682D4)](https://doi.org/10.5281/zenodo.17574447)

## 🔴 **[PLAY / BUILD MARS SYNAPSE: RED GENESIS](game/)**

**[OPEN THE INTERACTIVE CST FIELD](index.html)** · **[READ CST 2026](docs/CST_2026.md)** · **[CHECK THE CLAIM LEDGER](docs/CLAIMS_AND_EVIDENCE.md)**

</div>

---

## The idea in one sentence

**CST asks what becomes possible when a system is modeled as a persistent field of interacting state whose components continuously influence one another through memory, signal, distance, recurrence, learned association, and network coupling.**

This repository is the origin layer of that work. It preserves the early cosmic simulator, the 11D/12D mathematical lineage, the Python and Unity experiments, the modern evidence boundaries, and now a complete playable interpretation of the architecture.

The point is not to ask people to believe a strange idea. The point is to make enough of the idea executable that people can inspect it, test it, play with it, break it, measure it, and improve it.

---

# MARS SYNAPSE // RED GENESIS

The old CST simulator could generate worlds and evolving entities. **Red Genesis turns that lineage into an actual game.**

You are no longer watching the simulation.

You land on Mars.

You move through it, scan it, mine it, survive it, build on it, research it, remember what happened, create settlements, connect those settlements, and try to stabilize a planetary network.

### Core loop

**Land → scan → mine → survive → build → research → expand → connect → adapt → activate Red Genesis.**

The browser game includes:

- deterministic seeded Mars worlds
- approximately 0.38g / 3.71 m/s² player-scale Mars gravity
- astronaut movement, sprinting and low-gravity jumping
- rover traversal, battery use and cargo
- oxygen, suit energy, radiation and thermal stress
- day/night solar behavior
- dust storms and solar-radiation events
- deterministic resource geology and anomaly scanning
- gathering and inventory
- construction and production
- power generation and storage
- water and oxygen processing
- research and technology unlocks
- research-gated advanced structures
- settlement founding and logistics
- CST synaptic links that strengthen through repeated local cooperation
- twelve-channel planetary state
- entropy and a clearly labeled CST `psiProxy`
- persistent event memory with previous-entry chaining
- five-act campaign progression
- achievements
- three manual save slots plus autosave
- victory and unrecoverable-colony failure states
- synthesized local game audio
- optional microphone-reactive cosmetic input
- desktop controls
- iPhone-sized touch controls
- reduced-motion mode
- high-contrast HUD
- text scaling and touch-control sensitivity

### Play locally

From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/game/
```

No Python CST backend, account, API key, CDN, or game engine is required for normal browser play.

Game documentation: [`game/README.md`](game/README.md)

Unity expansion contract: [`game/UNITY_BRIDGE.md`](game/UNITY_BRIDGE.md)

---

## Why CST changes the game instead of decorating it

The game maintains a normalized twelve-channel planetary vector:

| Channel | Gameplay meaning |
|---|---|
| D01 | Energy |
| D02 | Matter |
| D03 | Water |
| D04 | Atmosphere / life-support stability |
| D05 | Thermal resilience |
| D06 | Radiation resilience |
| D07 | Mobility |
| D08 | Knowledge |
| D09 | Memory |
| D10 | Settlement coherence |
| D11 | Network coupling |
| D12 | Adaptation |

Those values are recalculated from what actually exists in the colony: infrastructure, resources, environmental stress, research, remembered events, settlement health and network topology.

Research changes mechanics too. Rover research reduces battery consumption. Energy research increases storage capacity. Advanced habitats, extraction systems, automation and CST infrastructure require matching research instead of appearing as free cosmetic upgrades.

The conceptual update remains:

```text
state(t+1) = F(
  state(t),
  input(t),
  memory(t),
  neighbors(t),
  coupling(t),
  dynamics(t)
)
```

That is the bridge between the original CST idea and the playable system.

---

## What CST is

CST is strongest when three layers remain distinct.

| Layer | Meaning | Status |
|---|---|---|
| **Computational architecture** | Persistent state, coupled channels, memory, signal transforms, recurrent dynamics, affinity and routing | Implemented across the CST software family |
| **Simulation / game model** | Numerical entities and worlds evolving under declared rules | Implemented here |
| **Physical hypothesis** | A neural-network-like description may be useful for some physical/informational systems | Speculative and unproven |

The software can be useful even if the broad physical interpretation remains unverified.

---

## Original simulator lineage

The earlier repository already contained important pieces of the DNA used by Red Genesis.

### Python

- `cst_engine.py`
  - entity creation and evolution
  - legacy 11D position and velocity
  - 12-value internal memory
  - gravitational and distance-based interaction terms
  - entropy evolution
  - CST `psi` calculation
  - frequency-to-light mapping
  - CSV/JSON memory-node logging
  - SHA-256 previous-token chaining
- `ecosystem_engine.py`
  - repaired bounded ecosystem compatibility layer
- `socket_server.py`
  - local TCP transport for Unity-facing state
- `cst_functions.py`
  - later 12D informational-energy-density formulation

### Unity lineage

- `CSTClient.cs`
- `CosmicEngine.cs`
- `MicAnalyzer.cs`
- `MeshGenerator.cs`
- `PlanetFactory.cs`
- `ShaderGenerator.cs`
- `ProceduralMaterialGenerator.cs`
- `AtmosphereGenerator.cs`
- `MemoryRift.cs`
- `NPCBehavior.cs`

The original Unity files are preserved. Red Genesis does not require them to run, and the new player-scale Mars physics deliberately does not apply the old cosmic pairwise-force model directly to the astronaut or rover.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`game/UNITY_BRIDGE.md`](game/UNITY_BRIDGE.md).

---

## The CST equations

The repository preserves multiple historical stages rather than pretending the formulation never evolved.

### Legacy simulator potential

```text
psi = (phi*E + lambda*E*dt + L*m*c^2/scale + Omega*E/a0 + Ugrav) / V_11D
```

### Later 12D formulation

```text
psi_i = ( kinetic_i + synaptic_i + gravitational_i + informational_i ) / V_12D
```

The expanded historical expression remains in [`CST_Formula_Explanation.markdown`](CST_Formula_Explanation.markdown).

**The existence of twelve software coordinates is not evidence that physical spacetime literally has twelve dimensions.**

In Red Genesis, `psiProxy` is explicitly a gameplay proxy. It is not presented as a newly measured field.

---

## Run the original simulator

```bash
git clone https://github.com/NavisWORLD/The-theory-of-CST.git
cd The-theory-of-CST
python -m venv .venv
```

Activate the environment, then:

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

Mock-audio mode:

```bash
MOCK_AUDIO=1 python socket_server.py
```

Default local endpoint:

```text
127.0.0.1:5555
```

Protocol examples:

```text
ping
update
audio {"rms":0.42,"pitch":440.0}
```

---

## Verification

Python CST verification:

```bash
pip install -r requirements-dev.txt
MOCK_AUDIO=1 PYTHONPATH=. pytest -q
```

Red Genesis verification:

```bash
cd game
npm test
npm run check
```

GitHub Actions contains separate verification for the research runtime and the browser game.

---

## Evidence rule

| Label | Meaning |
|---|---|
| **IMPLEMENTED** | Code exists and can be inspected |
| **OBSERVED** | A captured runtime shows execution |
| **MEASURED** | A declared experiment produced a metric |
| **NULL** | A test failed its declared success criterion |
| **HYPOTHESIS** | A falsifiable proposition awaiting evidence |
| **MODEL / METAPHOR** | Useful conceptual language that is not itself literal physics or biology |

Repository claim ledger: [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md)

---

## Scientific boundary

This repository does **not** establish that:

- the universe is literally a biological neural network
- simulation theory has been proven
- twelve physical dimensions have been experimentally discovered
- the golden ratio is a fundamental law of nature
- CST's informational term is a newly measured physical force
- persistent software state is consciousness
- game `psiProxy` values are physical observations
- software variables automatically become physical observables

What it establishes is narrower and testable: there are executable attempts to turn a speculative idea into explicit state, memory, equations, signal transforms, network rules, experiments, simulations and now gameplay.

---

## Wider CST stack

### CST Libraries

[`NavisWORLD/Python-cst-libraries-`](https://github.com/NavisWORLD/Python-cst-libraries-)

Reusable work around persistent state, synaptic affinity, Hebbian-style association, memory, routing, model adapters and cross-language computation.

### COSMOS / CST Universe Manual

[`NavisWORLD/Volume-I-The-COSMOS-CST-Universe-Manual.`](https://github.com/NavisWORLD/Volume-I-The-COSMOS-CST-Universe-Manual.)

Public manual and reproducible memory architecture.

### Foundational deposit

**Cory Shane Davis, _12-Dimensional Cosmic Synapse Theory_, Zenodo**

DOI: **10.5281/zenodo.17574447**

A DOI is a persistent scholarly reference. It does not by itself validate a physical hypothesis or establish patent rights.

---

## Repository map

```text
.
├── README.md
├── index.html                         CST living-field visualization
├── game/
│   ├── index.html                     Mars Synapse: Red Genesis
│   ├── src/                           game systems and browser adapters
│   ├── tests/                         Node behavior suite
│   ├── schema/                        versioned game-state contract
│   ├── README.md                      player/developer guide
│   └── UNITY_BRIDGE.md                3D expansion contract
├── cst_engine.py                      legacy Python simulator
├── cst_functions.py                   later 12D formulation
├── ecosystem_engine.py                bounded ecosystem compatibility
├── socket_server.py                   local bridge
├── *.cs                               Unity simulation lineage
├── docs/
│   ├── CST_2026.md
│   ├── ARCHITECTURE.md
│   └── CLAIMS_AND_EVIDENCE.md
├── tests/                              Python verification
├── CITATION.cff
└── .github/workflows/
```

---

## Rights and reuse

Repository-level rights guidance is in [`CORY_DAVIS_IP_AND_ACCESS_NOTICE.md`](CORY_DAVIS_IP_AND_ACCESS_NOTICE.md).

This repository contains material from different points in its history, including files that may carry earlier file-specific license text. The rights notice preserves third-party rights and rights validly granted under earlier licenses. Inspect the specific file and history before assuming a reuse right.

---

## Support Cosmic Synapse

I create because I love creating: music, painting, software, writing, research, science, educational tools and experiments that make people ask better questions.

I want the work to remain as inspectable, teachable and shareable as responsibly possible. Protection is not the same as hoarding. We share one world, and I would rather give people tools to learn, build, test and imagine than lock everything away.

This is for anyone who **dares to wonder, dares to love, and dares to question the questions**, no matter what they believe. ❤️

### ☕ [Support Cosmic Synapse on Buy Me a Coffee](https://buymeacoffee.com/cosmic_syanpse)

Support is optional. Curiosity is free.

---

<div align="center">

### CST is strongest when the strange idea is made measurable.

**Read the code. Run the model. Play the world. Break the assumptions. Keep the evidence.**

</div>
