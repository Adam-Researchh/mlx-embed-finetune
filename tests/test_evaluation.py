import json
import sys

import pytest

import evaluate


def test_report_cannot_overwrite_input_pairs(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.jsonl"
    content = '{"query":"q","positive":"p"}\n'
    pairs.write_text(content)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--base-model", "unused",
                                     "--eval-pairs", str(pairs), "--json-out", str(pairs)])
    monkeypatch.setattr(evaluate, "load_encoder", lambda *a: pytest.fail("Must reject output before loading"))
    with pytest.raises(SystemExit):
        evaluate.main()
    assert pairs.read_text() == content


@pytest.mark.parametrize("metadata", [[], {"query_prefix": 42}, {"doc_prefix": None}])
def test_malformed_prompt_metadata_is_not_silently_ignored(tmp_path, metadata):
    (tmp_path / "training_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata"):
        evaluate.read_trained_prefixes(str(tmp_path))


def test_corrupt_prompt_metadata_fails(tmp_path):
    (tmp_path / "training_metadata.json").write_text('{"query_prefix":')
    with pytest.raises(ValueError, match="metadata"):
        evaluate.read_trained_prefixes(str(tmp_path))
