import json
from pathlib import Path

from app import PRONUNCIATIONS, app, confidence_cubed, needleman_wunsch_alignment, normalize_bg, parse_paragraphs


def test_parse_targets_extracts_markers():
    text = "I record[V] data and keep a record[N].\n\nWe object[V] to that object[N]."
    parsed = parse_paragraphs(text)
    assert len(parsed) == 2
    assert parsed[0]["tokens"][1] == "record"
    assert parsed[0]["targets"][0]["label"] == "V"
    assert parsed[0]["targets"][1]["label"] == "N"
    assert parsed[0]["targets"][0]["expected_stress"] == 2
    assert parsed[0]["targets"][1]["expected_stress"] == 1


def test_alignment_maps_exact_matches_with_insertions():
    ref = ["i", "record", "data"]
    hyp = ["i", "really", "record", "data"]
    mapping = needleman_wunsch_alignment(ref, hyp)
    assert mapping[0] == 0
    assert mapping[1] == 2
    assert mapping[2] == 3


def test_confidence_cubed_and_bg_norm_are_deterministic():
    c = confidence_cubed(0.5)
    assert c == 0.125
    assert normalize_bg(c) == 0.125
    assert normalize_bg(confidence_cubed(3.2)) == 1.0
    assert normalize_bg(confidence_cubed(-1)) == 0.0



def test_api_analyze_rejects_non_numeric_paragraph_id():
    client = app.test_client()
    resp = client.post(
        "/api/analyze",
        data={"paragraph_id": "foo"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"] == "paragraph_id must be 1..5"
    assert payload["request_id"]


def test_pronunciation_inventory_is_loaded_for_targets():
    assert PRONUNCIATIONS.get("content")
    assert PRONUNCIATIONS.get("increase")


def test_api_analyze_persists_hst_sidecar_and_wav_for_paragraph3_fixture(monkeypatch, tmp_path):
    def fake_analyze_payload(paragraph, wav_bytes):
        return {
            "deepgram_words": [],
            "alignment": [],
            "targets": [
                {
                    "token_index": 1,
                    "word": "record",
                    "label": "N",
                    "expected_stress": 1,
                    "inferred_stress": 1,
                    "correct": True,
                    "status": "ok",
                    "core_durations": {"syll1": 0.12, "syll2": 0.08},
                    "core_phones": {"syll1": "EH", "syll2": "AO"},
                    "deepgram_word_index": 1,
                    "deepgram_confidence": 0.9,
                    "deepgram_confidence_cubed": 0.729,
                    "feedback": "Good stress placement.",
                }
            ],
            "score_summary": {
                "percent_correct": 100.0,
                "total_targets": 1,
                "scored_targets": 1,
                "missing_targets": 0,
                "unaligned_targets": 0,
            },
            "render_words": [],
        }

    monkeypatch.setattr("app.analyze_payload", fake_analyze_payload)
    monkeypatch.setattr("app.BUCKET_DIR", str(tmp_path))

    wav_path = Path(__file__).parent / "abc340c7-fc39-41f0-b1a6-3557f83b7707.wav"
    wav_bytes = wav_path.read_bytes()

    client = app.test_client()
    with wav_path.open("rb") as fh:
        resp = client.post(
            "/api/analyze",
            data={
                "paragraph_id": "3",
                "native_exemplar": "true",
                "audio_wav": (fh, wav_path.name),
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "198.51.100.23, 35.191.0.1"},
        )
    assert resp.status_code == 200
    payload = resp.get_json()

    persistence = payload["persistence"]
    recording_id = persistence["recording_id"]
    assert len(recording_id) == 18
    assert persistence["wav_path"].endswith(f"{recording_id}.wav")
    assert persistence["json_path"].endswith(f"{recording_id}.json")

    stored_wav = Path(persistence["wav_path"])
    stored_json = Path(persistence["json_path"])
    assert stored_wav.read_bytes() == wav_bytes

    sidecar = json.loads(stored_json.read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 1
    assert sidecar["recording_id"] == recording_id
    assert sidecar["timezone"] == "Pacific/Honolulu"
    assert sidecar["created_at_hst"].endswith("-10:00")
    assert sidecar["paragraph_id"] == 3
    assert sidecar["native_exemplar"] is True
    assert sidecar["request_ip"] == "198.51.100.23"
    assert sidecar["audio"]["bytes"] == len(wav_bytes)
    assert sidecar["targets"][0]["word_norm"] == "record"
    assert sidecar["targets"][0]["duration_ratio"] == 1.5
