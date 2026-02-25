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
