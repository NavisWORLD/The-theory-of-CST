# Red Genesis Unity Bridge Contract

The browser game is the playable reference implementation. This contract maps its state and rules onto the existing Unity/Python CST lineage without making the legacy runtime a requirement for browser play.

## Existing lineage

- `cst_engine.py`: entity state, entropy, CST psi, memory-node logging, ecosystem export.
- `CosmicEngine.cs`: Unity CST rendering, rigidbodies, cosmic entity interaction, audio reactivity, memory-rift activation.
- `PlanetFactory.cs`: generated terrain, ecosystems, atmospheres, and procedural materials.
- `MemoryRift.cs`: visual memory-state effects.
- `NPCBehavior.cs`: generated-life behavior layer.

## Browser → Unity mapping

| Browser state | Unity target |
|---|---|
| `player` | `MarsPlayerController` and rover controller |
| `environment` | `MarsEnvironmentController`, sky, lighting and hazard systems |
| `structures[]` | Buildable prefabs with production/network components |
| `settlements[]` | Settlement-root GameObjects |
| `cst.channels[12]` | `CSTPlanetaryState` component |
| `cst.entropy` | Bounded instability input for VFX and gameplay |
| `cst.psiProxy` | Explicit gameplay-proxy input for relay VFX/game rules |
| `cst.memory[]` | `MemoryRift` and chronicle/save adapter |
| `cst.links[]` | Relay/logistics edge components |
| `seed` | Deterministic procedural Mars seed |

## Local Mars physics

Planetary play should use Mars local gravity:

```csharp
Physics.gravity = new Vector3(0f, -3.71f, 0f);
```

Do not apply the legacy cosmic pairwise-force model directly to the astronaut or rover. Keep that model for the cosmic simulator. Red Genesis player-scale physics should use local Mars gravity, rigidbody collisions, rover suspension/traction, and game-scaled environmental hazards.

## CST boundary

CST affects gameplay through normalized state, network coupling, memory, entropy, and the `psiProxy` gameplay value. Unity may use those values for shaders, audio, event probability, efficiency modifiers, relay behavior and endgame readiness. They must remain visibly separate from claims of direct physical measurement.

## Serialization contract

Unity import/export should preserve `schemaVersion`, `seed`, `time`, `player`, `inventory`, `resources`, `structures`, `settlements`, `research`, `missions`, `achievements`, `cst`, `environment`, `stats`, `flags`, and `settings`. The browser save schema in `schema/game-state.schema.json` is the canonical interchange shape for version 1.
