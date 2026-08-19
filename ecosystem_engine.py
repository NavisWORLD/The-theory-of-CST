"""Lightweight ecosystem state used by the legacy CST universe simulator.

The engine is intentionally a computational toy. It supplies the interface that
``cst_engine.py`` already expects without claiming biological realism.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class EcosystemState:
    entity_id: int
    vitality: float
    complexity: float
    resource_flux: float
    climate_index: float
    age_seconds: float = 0.0

    def export(self) -> Dict[str, float]:
        return asdict(self)


class EcosystemEngine:
    """Maintain bounded, inspectable toy ecosystem state for planet entities."""

    PLANET_TYPE = 1

    def __init__(self) -> None:
        self._states: Dict[int, EcosystemState] = {}

    def add_ecosystem(self, entity: Any) -> Optional[Dict[str, float]]:
        if int(getattr(entity, "entity_type", -1)) != self.PLANET_TYPE:
            return None

        entity_id = int(entity.id)
        entropy = _clamp01(getattr(entity, "entropy", 0.5))
        level = _clamp01(getattr(entity, "ecosystem_level", 0.0))
        freq = max(20.0, min(20000.0, float(getattr(entity, "freq", 440.0))))
        freq_norm = (freq - 20.0) / (20000.0 - 20.0)

        state = EcosystemState(
            entity_id=entity_id,
            vitality=_clamp01(0.55 * level + 0.45 * entropy),
            complexity=_clamp01(0.65 * level + 0.35 * (1.0 - abs(0.5 - entropy) * 2.0)),
            resource_flux=_clamp01(0.35 + 0.4 * entropy + 0.25 * freq_norm),
            climate_index=_clamp01(1.0 - abs(freq_norm - 0.35) * 1.4),
        )
        self._states[entity_id] = state
        return state.export()

    def update(self, entities: Iterable[Any], audio_data: Optional[dict], dt: float) -> None:
        dt = max(0.0, min(float(dt), 10.0))
        if dt == 0.0:
            return

        rms = 0.0
        pitch_norm = 0.0
        if isinstance(audio_data, dict):
            try:
                rms = _clamp01(audio_data.get("rms", 0.0))
            except (TypeError, ValueError):
                rms = 0.0
            try:
                freqs = audio_data.get("freqs")
                pitch = float(freqs[0]) if freqs is not None and len(freqs) else 440.0
                pitch_norm = _clamp01((pitch - 20.0) / (20000.0 - 20.0))
            except (TypeError, ValueError, IndexError):
                pitch_norm = 0.0

        for entity in entities:
            if int(getattr(entity, "entity_type", -1)) != self.PLANET_TYPE:
                continue

            entity_id = int(entity.id)
            if entity_id not in self._states:
                self.add_ecosystem(entity)
            state = self._states[entity_id]

            entropy = _clamp01(getattr(entity, "entropy", 0.5))
            current_level = _clamp01(getattr(entity, "ecosystem_level", 0.0))

            # Small, bounded evolution terms keep the legacy visual simulation moving.
            drive = 0.45 * entropy + 0.35 * rms + 0.20 * pitch_norm
            target = _clamp01(0.65 * current_level + 0.35 * drive)
            step = min(1.0, dt * 0.08)
            next_level = _clamp01(current_level + (target - current_level) * step)
            entity.ecosystem_level = next_level

            state.vitality = _clamp01(state.vitality + (next_level - state.vitality) * step)
            state.complexity = _clamp01(state.complexity + ((entropy + next_level) / 2.0 - state.complexity) * step)
            state.resource_flux = _clamp01(0.65 * state.resource_flux + 0.35 * drive)
            state.climate_index = _clamp01(0.7 * state.climate_index + 0.3 * (1.0 - abs(pitch_norm - 0.35)))
            state.age_seconds = round(state.age_seconds + dt, 9)

    def export(self, entity_id: int) -> Optional[Dict[str, float]]:
        state = self._states.get(int(entity_id))
        return None if state is None else state.export()
