from types import SimpleNamespace

from ecosystem_engine import EcosystemEngine


def planet(entity_id=7, entropy=0.6, freq=1400.0, level=0.25):
    return SimpleNamespace(
        id=entity_id,
        entity_type=1,
        entropy=entropy,
        freq=freq,
        ecosystem_level=level,
    )


def test_add_ecosystem_registers_planet_and_exports_bounded_state():
    engine = EcosystemEngine()
    p = planet()

    state = engine.add_ecosystem(p)
    exported = engine.export(p.id)

    assert exported == state
    assert exported["entity_id"] == p.id
    assert 0.0 <= exported["vitality"] <= 1.0
    assert 0.0 <= exported["complexity"] <= 1.0
    assert 0.0 <= exported["resource_flux"] <= 1.0
    assert 0.0 <= exported["climate_index"] <= 1.0


def test_update_tracks_planet_level_and_audio_without_creating_non_planets():
    engine = EcosystemEngine()
    p = planet(level=0.4)
    star = SimpleNamespace(id=2, entity_type=0, entropy=0.4, freq=700.0, ecosystem_level=0.0)
    engine.add_ecosystem(p)

    engine.update([p, star], {"rms": 0.8, "freqs": [880.0]}, 0.5)

    exported = engine.export(p.id)
    assert exported is not None
    assert exported["age_seconds"] == 0.5
    assert 0.0 <= p.ecosystem_level <= 1.0
    assert engine.export(star.id) is None


def test_export_returns_none_for_unknown_entity():
    assert EcosystemEngine().export(9999) is None


def test_update_accepts_numpy_frequency_arrays_used_by_cst_engine():
    import numpy as np

    engine = EcosystemEngine()
    p = planet()
    engine.add_ecosystem(p)

    engine.update([p], {"rms": 0.4, "freqs": np.array([440.0])}, 0.1)

    assert engine.export(p.id)["age_seconds"] == 0.1


def test_numpy_pitch_changes_climate_response():
    import numpy as np

    low = EcosystemEngine()
    high = EcosystemEngine()
    p1 = planet(entity_id=11)
    p2 = planet(entity_id=12)
    low.add_ecosystem(p1)
    high.add_ecosystem(p2)

    low.update([p1], {"rms": 0.2, "freqs": np.array([220.0])}, 1.0)
    high.update([p2], {"rms": 0.2, "freqs": np.array([12000.0])}, 1.0)

    assert low.export(p1.id)["climate_index"] != high.export(p2.id)["climate_index"]
