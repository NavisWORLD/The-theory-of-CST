import { GAME_SCHEMA_VERSION } from './constants.js';
import { isValidStateShape } from './state.js';

const PREFIX='mars-synapse:red-genesis';
const slotKey=slot=>`${PREFIX}:slot:${Math.max(1,Math.min(3,Number(slot)||1))}`;
export const AUTOSAVE_KEY=`${PREFIX}:autosave`;

export function serializeGame(state){
  return JSON.stringify({savedAt:new Date().toISOString(),schemaVersion:GAME_SCHEMA_VERSION,state});
}

export function deserializeGame(text){
  try{
    const parsed=JSON.parse(text);
    const state=parsed?.state ?? parsed;
    if((parsed?.schemaVersion ?? state?.schemaVersion)!==GAME_SCHEMA_VERSION) return null;
    return isValidStateShape(state)?state:null;
  }catch{return null;}
}

export function saveSlot(storage,state,slot=1){
  if(!storage?.setItem || !isValidStateShape(state)) return false;
  storage.setItem(slotKey(slot),serializeGame(state));
  return true;
}

export function loadSlot(storage,slot=1){
  if(!storage?.getItem) return null;
  const raw=storage.getItem(slotKey(slot));
  return raw?deserializeGame(raw):null;
}

export function deleteSlot(storage,slot=1){
  storage?.removeItem?.(slotKey(slot));
}

export function saveAutosave(storage,state){
  if(!storage?.setItem || !isValidStateShape(state)) return false;
  storage.setItem(AUTOSAVE_KEY,serializeGame(state));
  return true;
}

export function loadAutosave(storage){
  const raw=storage?.getItem?.(AUTOSAVE_KEY);
  return raw?deserializeGame(raw):null;
}

export function getSlotSummary(storage,slot=1){
  const state=loadSlot(storage,slot);
  if(!state) return null;
  return {slot:Number(slot),seed:state.seed,sol:state.time.sol,playtime:state.stats.playtime,settlements:state.settlements.length,victory:!!state.flags.victory};
}
