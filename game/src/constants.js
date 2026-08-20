export const GAME_SCHEMA_VERSION = 1;
export const MARS_GRAVITY = 3.71;
export const SOL_SECONDS = 720;
export const WORLD_RADIUS = 2400;
export const PLAYER = Object.freeze({
  walkAcceleration: 32,
  sprintAcceleration: 48,
  roverAcceleration: 62,
  roverMaxSpeed: 42,
  footMaxSpeed: 11,
  drag: 5.4,
  jumpVelocity: 7.4,
  gatherCooldown: 0.45,
});

export const RESOURCE_TYPES = ['ice','regolith','metal','silica','carbon','salvage','anomaly'];

export const BUILDINGS = Object.freeze({
  habitat: { cost: { regolith: 10, metal: 6, silica: 3 }, power: -1, radius: 24, label: 'Habitat' },
  solar: { cost: { metal: 5, silica: 4 }, power: 7, radius: 18, label: 'Solar Array' },
  battery: { cost: { metal: 4, salvage: 2 }, power: -0.2, radius: 16, label: 'Battery Bank' },
  oxygen: { cost: { metal: 5, silica: 2 }, power: -2.2, radius: 18, label: 'Oxygen Extractor' },
  water: { cost: { metal: 5, silica: 2 }, power: -2, radius: 18, label: 'Water Processor' },
  storage: { cost: { regolith: 4, metal: 3 }, power: -0.1, radius: 18, label: 'Storage' },
  fabricator: { cost: { metal: 8, silica: 3, salvage: 4 }, power: -3, radius: 20, label: 'Fabricator' },
  relay: { cost: { metal: 4, silica: 3, salvage: 2 }, power: -1.4, radius: 16, label: 'Comm Relay' },
  'rover-bay': { cost: { metal: 10, salvage: 4 }, power: -2, radius: 26, label: 'Rover Bay' },
  research: { cost: { metal: 7, silica: 5, salvage: 5 }, power: -3, radius: 20, label: 'Research Station' },
  radiator: { cost: { metal: 4, silica: 2 }, power: -0.8, radius: 18, label: 'Thermal Controller' },
  shelter: { cost: { regolith: 12, metal: 4 }, power: -0.4, radius: 24, label: 'Radiation Shelter' },
  greenhouse: { cost: { silica: 10, metal: 8, water: 4 }, power: -4, radius: 26, label: 'Greenhouse', advanced: true },
  reactor: { cost: { metal: 18, salvage: 10, anomaly: 1 }, power: 20, radius: 24, label: 'Aux Reactor', advanced: true },
  'deep-drill': { cost: { metal: 12, salvage: 4 }, power: -5, radius: 22, label: 'Deep Ice Drill', advanced: true },
  miner: { cost: { metal: 14, salvage: 6 }, power: -5, radius: 22, label: 'Automated Miner', advanced: true },
  'drone-pad': { cost: { metal: 12, silica: 4, salvage: 8 }, power: -4, radius: 24, label: 'Drone Pad', advanced: true },
  uplink: { cost: { metal: 10, silica: 8, salvage: 8 }, power: -4, radius: 20, label: 'Orbital Uplink', advanced: true },
  'cst-relay': { cost: { metal: 12, silica: 8, anomaly: 1 }, power: -4.5, radius: 18, label: 'CST Relay', advanced: true },
  'memory-archive': { cost: { metal: 8, silica: 8, salvage: 6 }, power: -2.5, radius: 20, label: 'Memory Archive', advanced: true },
  'lattice-hub': { cost: { metal: 18, silica: 12, anomaly: 2 }, power: -6, radius: 28, label: 'Synaptic Lattice Hub', advanced: true },
  'settlement-core': { cost: { regolith: 20, metal: 18, silica: 8, salvage: 6 }, power: -5, radius: 34, label: 'Settlement Core', advanced: true },
});

export const TECHS = Object.freeze({
  'survival-1': { label: 'Emergency Life Support', cost: 0 },
  'mobility-1': { label: 'Rover Efficiency', cost: 8 },
  'energy-1': { label: 'High Density Storage', cost: 10 },
  'extraction-1': { label: 'Deep Extraction', cost: 12 },
  'habitat-1': { label: 'Redundant Habitat', cost: 14 },
  'automation-1': { label: 'Repair Drones', cost: 16 },
  'science-1': { label: 'Anomaly Spectrometry', cost: 16 },
  'cst-1': { label: 'CST Relay Logic', cost: 20 },
  'cst-2': { label: 'Synaptic Lattice', cost: 28, requires: 'cst-1' },
});
