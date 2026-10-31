"""The Second Listen API (/api/v2) with the verifier and Whisper replaced by stand-ins (no data or model needed)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/ is not an installed package

from app import server  # noqa: E402

CID = "day1_consultation01"


class FakeVerifier:
    def list_consultations(self, model="small"):
        return [{"id": CID, "split": "test", "duration": 457.9, "n_turns": 2, "n_windows": 1, "has_audio": True}]

    def consultation_payload(self, consultation, unit="turn", model="small"):
        if consultation != CID:
            raise KeyError(consultation)
        return {"id": consultation, "split": "test", "duration": 457.9, "unit": unit, "units": []}

    def review_curves(self, model="small"):
        return {"turn": {"x_seconds": [0, 5]}, "window": {"x_seconds": [0, 5]}, "meta": {}}


def events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_verifier", lambda: FakeVerifier())
    monkeypatch.setattr(server, "_available", lambda m: m == "small")
    return TestClient(server.app)


def test_consultations_and_payload(client):
    assert client.get("/api/v2/consultations").json()[0]["id"] == CID
    r = client.get(f"/api/v2/consultation/{CID}", params={"unit": "window"})
    assert r.status_code == 200 and r.json()["unit"] == "window"


@pytest.mark.parametrize("path,status", [
    ("/api/v2/consultation/day9_consultation99", 404),     # well formed, not in the data
    ("/api/v2/consultation/..%2Fsecrets", 404),            # never reaches the filesystem
    (f"/api/v2/consultation/{CID}?unit=sentence", 422),
])
def test_payload_rejects_bad_requests(client, path, status):
    assert client.get(path).status_code == status


def test_untrained_verifier_is_a_clear_503(client, monkeypatch):
    class Untrained(FakeVerifier):
        def review_curves(self, *args, **kwargs):
            raise FileNotFoundError("results/tables/small/verifier/curves.json missing; run `asrshift verifier`")

        consultation_payload = review_curves

    monkeypatch.setattr(server, "_verifier", lambda: Untrained())
    for path in ("/api/v2/curves", f"/api/v2/consultation/{CID}"):
        r = client.get(path)
        assert r.status_code == 503 and "asrshift verifier" in r.json()["detail"]


def test_models(client, monkeypatch):
    r = client.get("/api/v2/models").json()
    assert r["default"] == "small" and r["live"] == "small"
    assert {m["id"]: m["available"] for m in r["models"]} == {"turbo-lora-mixed": False, "small": True}
    assert client.get(f"/api/v2/consultation/{CID}").json()["model"] == "small"
    assert client.get("/api/v2/curves", params={"model": "turbo-lora-mixed"}).status_code == 503
    assert client.get("/api/v2/curves", params={"model": "large-v2"}).status_code == 422
    monkeypatch.setattr(server, "_available", lambda m: True)   # the fine-tuned model's results present
    assert client.get("/api/v2/models").json()["default"] == "turbo-lora-mixed"
    assert client.get(f"/api/v2/consultation/{CID}").json()["model"] == "turbo-lora-mixed"


def test_curves(client):
    assert set(client.get("/api/v2/curves").json()) == {"turn", "window", "meta"}


def test_audio_supports_range_requests(client, monkeypatch, tmp_path):
    wav = tmp_path / "mix.wav"
    sf.write(wav, np.zeros(16_000, dtype="float32"), 16_000, subtype="PCM_16")
    monkeypatch.setattr(server, "_mix_path", lambda cid: wav)
    size = wav.stat().st_size
    full = client.get(f"/api/v2/consultation/{CID}/audio")
    assert full.status_code == 200 and len(full.content) == size
    part = client.get(f"/api/v2/consultation/{CID}/audio", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and len(part.content) == 100
    assert part.headers["content-range"] == f"bytes 100-199/{size}"
    tail = client.get(f"/api/v2/consultation/{CID}/audio", headers={"Range": "bytes=-10"})
    assert tail.status_code == 206 and tail.content == full.content[-10:]
    assert client.get(f"/api/v2/consultation/{CID}/audio", headers={"Range": f"bytes={size}-"}).status_code == 416


def fake_turns():
    return pd.DataFrame({
        "unit_id": [f"pm_turn_{CID}_doctor_001", f"pm_turn_{CID}_patient_001"], "dataset": "primock57",
        "subset": ["turn_doctor", "turn_patient"], "consultation": CID, "roles": ["doctor", "patient"],
        "start": [1.0, 4.0], "end": [3.5, 6.0], "split": "test",
    })


def test_live_streams_turns_in_time_order(client, monkeypatch):
    import asrshift.audio

    monkeypatch.setattr(server, "committed_turns", fake_turns)
    monkeypatch.setattr(asrshift.audio, "primock_audio", lambda src, a, b: np.zeros(int((b - a) * 16_000), "float32"))
    monkeypatch.setattr(server, "_whisper", lambda: (None, {"asr": {"decode": {}}}))
    seen = []

    def fake_decode(arr, offset, kind, decode_cfg, second, vad=False):
        seen.append((offset, second))
        word = {"text": "hello", "t0": offset, "t1": offset + 0.4, "p": 0.9, "score": 0.1, "critical": False}
        return [], {"words": [word], "gaps": [{"after": 0, "t": offset + 0.4, "score": 0.0, "suggestion": ""}]}, 0.01

    monkeypatch.setattr(server, "_decode_and_score", fake_decode)
    r = client.get(f"/api/v2/live/{CID}", params={"second": 1})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r.text)
    assert [e for e, _ in ev] == ["turn", "progress", "turn", "progress", "done"]
    turns = [d for e, d in ev if e == "turn"]
    assert [t["role"] for t in turns] == ["doctor", "patient"]
    assert turns[0]["words"][0]["t0"] == 1.0 and ev[3][1]["decoded_until"] == 6.0
    assert seen == [(1.0, True), (4.0, True)]


def test_new_live_run_cancels_older_ones(client, monkeypatch):
    started = []
    monkeypatch.setattr(server, "_run_live", lambda job, cid, second: (started.append(job), job.put("done", {})))
    client.get(f"/api/v2/live/{CID}")
    client.get(f"/api/v2/live/{CID}")
    assert len(started) == 2 and started[0].cancelled.is_set() and not started[1].cancelled.is_set()
    assert server._live_jobs == [started[1]]


def test_live_reports_missing_consultation(client, monkeypatch):
    monkeypatch.setattr(server, "committed_turns", fake_turns)
    ev = events(client.get("/api/v2/live/day9_consultation99").text)
    assert ev == [("error", {"message": "no turns for day9_consultation99"})]


def test_upload_creates_a_job_with_replayable_events(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)

    def fake_run(job, path, second):
        assert path.exists()
        job.put("turn", {"id": "seg_0000", "role": "speaker", "start": 0.0, "end": 1.0, "words": [], "gaps": []})
        job.put("done", {"audio_s": 1.0})

    monkeypatch.setattr(server, "_run_upload", fake_run)
    r = client.post("/api/v2/upload", files={"file": ("note.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 200
    job = r.json()["job"]
    for _ in range(2):  # a second client gets the same events
        ev = events(client.get(f"/api/v2/jobs/{job}/events").text)
        assert [e for e, _ in ev] == ["turn", "done"]
    assert client.post("/api/v2/upload", files={"file": ("notes.txt", b"hi", "text/plain")}).status_code == 415
    assert client.get("/api/v2/jobs/nope/events").status_code == 404


def test_v1_endpoints_still_mounted(client):
    paths = {r.path for r in server.app.routes}
    assert {"/api/meta", "/api/summary", "/api/queue", "/api/unit/{unit_id}", "/api/transcribe/{unit_id}"} <= paths
