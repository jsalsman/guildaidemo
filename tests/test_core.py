import base64
import json
import time
from pathlib import Path

from app import (
    PRONUNCIATIONS,
    load_adaptive_thresholds,
    DeepgramAPIError,
    app,
    confidence_cubed,
    gaussian_pdf,
    get_learned_threshold,
    infer_stress_from_word,
    needleman_wunsch_alignment,
    normalize_bg,
    parse_paragraphs,
)


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




def test_gaussian_pdf_nonpositive_sigma_returns_zero():
    assert gaussian_pdf(0.0, 0.0, 0.0) == 0.0
    assert gaussian_pdf(1.0, 0.0, -1.0) == 0.0

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
    assert payload["error"] == "paragraph_id must be 1..10"
    assert payload["request_id"]


def test_health_endpoints_return_ok_payload():
    client = app.test_client()

    for route in ("/healthz", "/api/healthz"):
        resp = client.get(route)
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["status"] == "ok"
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
    assert len(recording_id) == 19
    assert recording_id.endswith("e")
    assert persistence["wav_path"].endswith(f"{recording_id}.wav")
    assert persistence["json_path"].endswith(f"{recording_id}.json")

    stored_wav = Path(persistence["wav_path"])
    stored_json = Path(persistence["json_path"])
    deadline = time.time() + 3.0
    while time.time() < deadline and (not stored_wav.exists() or not stored_json.exists()):
        time.sleep(0.01)

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


def test_threshold_learning_builds_gaussian_intersection_boundary(tmp_path):
    sidecars = [
        {"native_exemplar": True, "paragraph_id": 1, "targets": [
            {"status": "ok", "word_norm": "record", "token_index": 2, "expected_stress": 1, "duration_ratio_log": 0.6, "core_durations": {"syll1": 0.12, "syll2": 0.08}},
            {"status": "ok", "word_norm": "record", "token_index": 2, "expected_stress": 1, "duration_ratio_log": 0.4, "core_durations": {"syll1": 0.11, "syll2": 0.09}},
            {"status": "ok", "word_norm": "record", "token_index": 2, "expected_stress": 2, "duration_ratio_log": -0.5, "core_durations": {"syll1": 0.08, "syll2": 0.12}},
            {"status": "ok", "word_norm": "record", "token_index": 2, "expected_stress": 2, "duration_ratio_log": -0.3, "core_durations": {"syll1": 0.09, "syll2": 0.11}},
            {"status": "ok", "word_norm": "record", "token_index": 2, "expected_stress": 1, "duration_ratio_log": 0.5, "core_durations": {"syll1": 0.12, "syll2": 0.08}},
            {"status": "ok", "word_norm": "record", "token_index": 2, "expected_stress": 2, "duration_ratio_log": -0.4, "core_durations": {"syll1": 0.08, "syll2": 0.12}},
        ]},
    ]
    for i, data in enumerate(sidecars):
        (tmp_path / f"{i}.json").write_text(json.dumps(data), encoding="utf-8")

    thresholds = load_adaptive_thresholds(str(tmp_path))
    assert thresholds["word:record"]["threshold"] == 0.05
    assert thresholds["ctx:record:1:2"]["threshold"] == 0.05
    assert thresholds["word:record"]["class1_count"] == 3
    assert thresholds["word:record"]["class2_count"] == 3


def test_threshold_learning_falls_back_when_insufficient_exemplars(tmp_path):
    data = {
        "native_exemplar": True,
        "targets": [
            {"status": "ok", "word_norm": "record", "expected_stress": 1, "duration_ratio_log": 0.5, "core_durations": {"syll1": 0.11, "syll2": 0.08}},
            {"status": "ok", "word_norm": "record", "expected_stress": 2, "duration_ratio_log": -0.5, "core_durations": {"syll1": 0.08, "syll2": 0.11}},
        ],
    }
    (tmp_path / "one.json").write_text(json.dumps(data), encoding="utf-8")
    thresholds = load_adaptive_thresholds(str(tmp_path))
    assert "word:record" not in thresholds


def test_infer_stress_uses_learned_threshold(monkeypatch, tmp_path):
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")
    trained = {
        "native_exemplar": True,
        "paragraph_id": 3,
        "targets": [
            {"status": "ok", "word_norm": "record", "token_index": 1, "expected_stress": 1, "duration_ratio_log": 0.6, "core_durations": {"syll1": 0.12, "syll2": 0.08}},
            {"status": "ok", "word_norm": "record", "token_index": 1, "expected_stress": 1, "duration_ratio_log": 0.5, "core_durations": {"syll1": 0.12, "syll2": 0.08}},
            {"status": "ok", "word_norm": "record", "token_index": 1, "expected_stress": 2, "duration_ratio_log": -0.6, "core_durations": {"syll1": 0.08, "syll2": 0.12}},
            {"status": "ok", "word_norm": "record", "token_index": 1, "expected_stress": 2, "duration_ratio_log": -0.5, "core_durations": {"syll1": 0.08, "syll2": 0.12}},
            {"status": "ok", "word_norm": "record", "token_index": 1, "expected_stress": 1, "duration_ratio_log": 0.55, "core_durations": {"syll1": 0.12, "syll2": 0.08}},
            {"status": "ok", "word_norm": "record", "token_index": 1, "expected_stress": 2, "duration_ratio_log": -0.55, "core_durations": {"syll1": 0.08, "syll2": 0.12}},
        ],
    }
    (tmp_path / "trained.json").write_text(json.dumps(trained), encoding="utf-8")

    monkeypatch.setattr("app.BUCKET_DIR", str(tmp_path))

    phones = [
        {"phone": "R", "start": 0.00, "end": 0.01},
        {"phone": "EH", "start": 0.01, "end": 0.08},
        {"phone": "K", "start": 0.08, "end": 0.09},
        {"phone": "ER", "start": 0.09, "end": 0.14},
    ]
    thresholds = load_adaptive_thresholds(str(tmp_path))
    result = infer_stress_from_word("record", phones, thresholds, paragraph_id=3, token_index=1)
    assert result["decision_method"] == "learned_threshold"
    assert result["learned_threshold"] is not None
    assert result["threshold_key"] == "ctx:record:3:1"
    assert result["inferred_stress"] == 1
    assert result["decision_confidence"] is not None




def test_infer_stress_naive_has_no_decision_confidence():
    phones = [
        {"phone": "R", "start": 0.00, "end": 0.01},
        {"phone": "EH", "start": 0.01, "end": 0.05},
        {"phone": "K", "start": 0.05, "end": 0.06},
        {"phone": "ER", "start": 0.06, "end": 0.10},
    ]
    result = infer_stress_from_word("record", phones, {})
    assert result["decision_method"] == "naive_duration"
    assert result["decision_confidence"] is None

def test_malformed_json_ignored_for_threshold_cache(monkeypatch, tmp_path):
    (tmp_path / "bad.json").write_text("}", encoding="utf-8")
    monkeypatch.setattr("app.BUCKET_DIR", str(tmp_path))
    thresholds = load_adaptive_thresholds(str(tmp_path))
    assert get_learned_threshold(thresholds, "record") is None



def test_api_analyze_surfaces_friendly_deepgram_auth_error(monkeypatch, tmp_path):
    def fake_analyze_payload(paragraph, wav_bytes):
        raise DeepgramAPIError(
            "Deepgram rejected this API key (401 Unauthorized). Please set a working API key and submit again.",
            status_code=401,
        )

    monkeypatch.setattr("app.analyze_payload", fake_analyze_payload)

    wav_path = Path(__file__).parent / "abc340c7-fc39-41f0-b1a6-3557f83b7707.wav"
    client = app.test_client()
    with wav_path.open("rb") as fh:
        resp = client.post(
            "/api/analyze",
            data={"paragraph_id": "1", "audio_wav": (fh, wav_path.name)},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 401
    payload = resp.get_json()
    assert payload["error_code"] == 401
    assert "set a working API key" in payload["error"]
    assert "https://api.deepgram.com" not in payload["error"]



def test_a2a_bad_media_maps_to_invalid_params_error(monkeypatch):
    def fake_analyze_payload(paragraph, wav_bytes, deepgram_api_key=None):
        raise DeepgramAPIError(
            "Deepgram could not process this audio request (400 Bad Request). Please verify the media format and try again.",
            status_code=400,
        )

    monkeypatch.setattr("app.analyze_payload", fake_analyze_payload)

    wav_path = Path(__file__).parent / "abc340c7-fc39-41f0-b1a6-3557f83b7707.wav"
    b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")

    client = app.test_client()
    resp = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "bad-media",
            "method": "pronunciation.evaluate",
            "params": {"paragraph_id": 1, "audio_wav_base64": b64},
        },
    )

    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"]["code"] == -32602
    assert "verify the media format" in payload["error"]["message"]


def test_a2a_surfaces_friendly_deepgram_rate_limit_error(monkeypatch):
    def fake_analyze_payload(paragraph, wav_bytes, deepgram_api_key=None):
        raise DeepgramAPIError(
            "Deepgram rate limit reached (429 Too Many Requests). Please wait briefly, then press Submit for Analysis again.",
            status_code=429,
        )

    monkeypatch.setattr("app.analyze_payload", fake_analyze_payload)

    wav_path = Path(__file__).parent / "abc340c7-fc39-41f0-b1a6-3557f83b7707.wav"
    b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")

    client = app.test_client()
    resp = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "rate-limit",
            "method": "pronunciation.evaluate",
            "params": {"paragraph_id": 1, "audio_wav_base64": b64},
        },
    )

    assert resp.status_code == 429
    payload = resp.get_json()
    assert payload["error"]["code"] == -32000
    assert "Too Many Requests" in payload["error"]["message"]


def test_a2a_client_can_read_model_card_and_submit_paragraph3_wav(monkeypatch):
    def fake_analyze_payload(paragraph, wav_bytes):
        return {
            "deepgram_words": [{"word": "record", "start": 0.1, "end": 0.4, "confidence": 0.97, "confidence_cubed": 0.912673}],
            "alignment": [{"ref_index": 1, "hyp_index": 0}],
            "targets": [{"word": "record", "status": "ok", "expected_stress": 1, "inferred_stress": 1, "correct": True}],
            "score_summary": {"percent_correct": 100.0, "total_targets": 7, "scored_targets": 7, "missing_targets": 0, "unaligned_targets": 0},
            "render_words": [],
        }

    def fake_persist_submission(**kwargs):
        return {
            "recording_id": "260227123456000001",
            "wav_path": "/bucket/260227123456000001.wav",
            "json_path": "/bucket/260227123456000001.json",
        }

    monkeypatch.setattr("app.analyze_payload", fake_analyze_payload)
    monkeypatch.setattr("app.persist_submission", fake_persist_submission)

    client = app.test_client()

    card_resp = client.get("/.well-known/agent.json")
    assert card_resp.status_code == 200
    card = card_resp.get_json()
    assert card["protocolVersion"] == "0.5.0"
    assert card["skills"][0]["name"] == "pronunciation.evaluate"
    assert card["capabilities"]["methods"]["pronunciation.evaluate"]["requiredParams"] == ["audio_wav_base64"]

    wav_path = Path(__file__).parent / "abc340c7-fc39-41f0-b1a6-3557f83b7707.wav"
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")

    rpc_resp = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "p3-a2a-test",
            "method": "pronunciation.evaluate",
            "params": {"paragraph_id": 3, "audio_wav_base64": audio_b64},
        },
    )
    assert rpc_resp.status_code == 200
    rpc = rpc_resp.get_json()
    assert rpc["jsonrpc"] == "2.0"
    assert rpc["id"] == "p3-a2a-test"
    assert rpc["result"]["analysis"]["score_summary"]["percent_correct"] == 100.0
    assert rpc["result"]["persistence"]["recording_id"] == "260227123456000001"


def test_a2a_paragraph_helpers_expose_count_and_plain_text():
    client = app.test_client()

    count_resp = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "p-count", "method": "paragraphs.count", "params": {}},
    )
    assert count_resp.status_code == 200
    count_rpc = count_resp.get_json()
    assert count_rpc["result"]["paragraph_count"] == len(client.get("/api/paragraphs").get_json()["paragraphs"])

    text_resp = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "p-text-1",
            "method": "paragraphs.get_text",
            "params": {"paragraph_id": 1},
        },
    )
    assert text_resp.status_code == 200
    text_rpc = text_resp.get_json()
    assert text_rpc["result"]["paragraph_id"] == 1
    assert "[" not in text_rpc["result"]["paragraph_text"]


def test_a2a_paragraph_text_rejects_invalid_id():
    client = app.test_client()
    resp = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "p-text-invalid",
            "method": "paragraphs.get_text",
            "params": {"paragraph_id": 0},
        },
    )
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"]["code"] == -32602
    assert payload["error"]["message"].startswith("paragraph_id must be an int in range 1..")
