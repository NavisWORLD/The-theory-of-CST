import json


def test_python_stack_imports():
    import cst_functions
    import socket_server

    assert callable(cst_functions.compute_psi_i)
    assert callable(socket_server.start_server)


def test_engine_starts_and_pings_in_mock_audio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOCK_AUDIO", "1")

    from cst_engine import CSTEngine

    engine = CSTEngine()
    try:
        payload = json.loads(engine.ping())
        assert payload["status"] == "ok"
        assert payload["entities"] == 10
    finally:
        engine.cleanup()
