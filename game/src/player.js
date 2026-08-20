import { MARS_GRAVITY, PLAYER, WORLD_RADIUS } from './constants.js';
import { sampleResource } from './world.js';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function axis(input) {
  const x = (input.right ? 1 : 0) - (input.left ? 1 : 0) + (Number(input.axisX) || 0);
  const y = (input.down ? 1 : 0) - (input.up ? 1 : 0) + (Number(input.axisY) || 0);
  const m = Math.hypot(x, y);
  return m > 1 ? [x / m, y / m] : [x, y];
}

export function toggleVehicle(state) {
  if (state.player.mode === 'foot') {
    if (state.player.rover.health > 0 && state.player.rover.battery > 0) state.player.mode = 'rover';
  } else {
    state.player.mode = 'foot';
  }
  return state.player.mode;
}

export function stepPlayer(state, input = {}, dt = 1 / 60) {
  const p = state.player;
  const [ax, ay] = axis(input);
  const rover = p.mode === 'rover';
  const sprint = !rover && !!input.sprint && p.energy > 2;
  const accel = rover ? PLAYER.roverAcceleration : sprint ? PLAYER.sprintAcceleration : PLAYER.walkAcceleration;
  const maxSpeed = rover ? PLAYER.roverMaxSpeed : PLAYER.footMaxSpeed * (sprint ? 1.35 : 1);

  p.vx += ax * accel * dt;
  p.vy += ay * accel * dt;
  const drag = Math.max(0, 1 - PLAYER.drag * dt);
  if (!ax) p.vx *= drag;
  if (!ay) p.vy *= drag;
  const speed = Math.hypot(p.vx, p.vy);
  if (speed > maxSpeed) {
    p.vx = (p.vx / speed) * maxSpeed;
    p.vy = (p.vy / speed) * maxSpeed;
  }

  const ox = p.x, oy = p.y;
  p.x += p.vx * dt;
  p.y += p.vy * dt;
  const radius = Math.hypot(p.x, p.y);
  if (radius > WORLD_RADIUS) {
    p.x = (p.x / radius) * WORLD_RADIUS;
    p.y = (p.y / radius) * WORLD_RADIUS;
    p.vx *= -0.2;
    p.vy *= -0.2;
  }
  state.stats.distance += Math.hypot(p.x - ox, p.y - oy);

  if (!rover && input.jump && p.altitude <= 0.0001) p.verticalVelocity = PLAYER.jumpVelocity;
  p.verticalVelocity -= MARS_GRAVITY * dt;
  p.altitude += p.verticalVelocity * dt;
  if (p.altitude <= 0) {
    p.altitude = 0;
    p.verticalVelocity = 0;
  }

  if (Math.abs(ax) + Math.abs(ay) > 0.01) p.facing = Math.atan2(ay, ax);
  if (sprint) p.energy = clamp(p.energy - 1.4 * dt, 0, 100);
  if (rover && speed > 0.5) p.rover.battery = clamp(p.rover.battery - (0.12 + speed * 0.007) * dt, 0, 100);
  if (rover && p.rover.battery <= 0) {
    p.vx *= 0.85;
    p.vy *= 0.85;
  }
  p.gatherCooldown = Math.max(0, p.gatherCooldown - dt);
}

function nearStructure(state, types, radius = 34) {
  return state.structures.some(s => types.includes(s.type) && s.health > 0 && Math.hypot(s.x - state.player.x, s.y - state.player.y) <= radius);
}

export function stepSurvival(state, dt = 1 / 60) {
  const p = state.player;
  if (p.mode === 'rover') {
    p.oxygen = clamp(p.oxygen - 0.025 * dt, 0, 100);
    p.energy = clamp(p.energy + 0.12 * dt, 0, 100);
  } else {
    const exertion = Math.hypot(p.vx, p.vy) / PLAYER.footMaxSpeed;
    p.oxygen = clamp(p.oxygen - (0.045 + exertion * 0.035) * dt, 0, 100);
    p.energy = clamp(p.energy - (0.018 + exertion * 0.02) * dt, 0, 100);
  }

  const sheltered = nearStructure(state, ['habitat', 'shelter'], 32);
  const thermalControl = nearStructure(state, ['habitat', 'radiator'], 38);
  const radiationRate = state.environment.radiation * (sheltered ? 0.08 : p.mode === 'rover' ? 0.45 : 1);
  p.radiation = clamp(p.radiation + radiationRate * 0.02 * dt, 0, 100);
  const cold = Math.max(0, (-45 - state.environment.temperature) / 60);
  p.temperatureStress = clamp(p.temperatureStress + (thermalControl ? -0.8 : cold * 0.55) * dt, 0, 100);

  if (nearStructure(state, ['habitat'], 26)) {
    if (state.resources.oxygen > 0) {
      const refill = Math.min(12 * dt, 100 - p.oxygen, state.resources.oxygen);
      p.oxygen += refill;
      state.resources.oxygen -= refill;
    }
    p.energy = clamp(p.energy + 9 * dt, 0, 100);
    p.health = clamp(p.health + 2 * dt, 0, 100);
  }

  let damage = 0;
  if (p.oxygen <= 0) damage += 5.5 * dt;
  if (p.temperatureStress > 82) damage += ((p.temperatureStress - 82) / 18) * 1.4 * dt;
  if (p.radiation > 75) damage += ((p.radiation - 75) / 25) * 0.9 * dt;
  p.health -= damage;

  if (p.health <= 0) recoverPlayer(state);
}

export function recoverPlayer(state) {
  const base = state.settlements[0] || { x: 0, y: 0 };
  const p = state.player;
  p.x = base.x;
  p.y = base.y;
  p.vx = 0;
  p.vy = 0;
  p.altitude = 0;
  p.verticalVelocity = 0;
  p.health = 45;
  p.oxygen = 55;
  p.energy = 45;
  p.temperatureStress = Math.max(0, p.temperatureStress - 35);
  p.radiation = Math.max(0, p.radiation - 8);
  p.recoveryCount += 1;
  state.stats.rescues += 1;
  for (const key of Object.keys(state.inventory)) state.inventory[key] = Math.floor(state.inventory[key] * 0.9);
}

export function gatherAtPlayer(state, world) {
  if (state.player.gatherCooldown > 0) return null;
  const sample = sampleResource(world, state.player.x, state.player.y);
  const amount = sample.amount;
  state.inventory[sample.type] = (state.inventory[sample.type] || 0) + amount;
  state.player.gatherCooldown = PLAYER.gatherCooldown;
  state.stats.gathered += amount;
  if (state.player.mode === 'rover') state.player.rover.cargo = Math.min(state.player.rover.maxCargo, state.player.rover.cargo + amount);
  return { ...sample, amount };
}
