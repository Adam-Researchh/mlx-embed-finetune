import json
import pytest

from benchmarks.prepare_beir import prepare
from retrieval import load_pairs


def test_beir_conversion_keeps_multiple_positives_and_provenance(tmp_path):
    source = tmp_path / "source"
    (source / "qrels").mkdir(parents=True)
    (source / "corpus.jsonl").write_text(
        '{"_id":"d1","title":"Title","text":"First"}\n'
        '{"_id":"d2","title":"","text":"Second"}\n')
    (source / "queries.jsonl").write_text('{"_id":"q1","text":"Question"}\n')
    (source / "qrels/test.tsv").write_text(
        'query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\n')
    out = tmp_path / "output"
    manifest = prepare(source, "test", out)
    assert manifest["queries"] == 1 and manifest["documents"] == 2
    assert len(manifest["source_sha256"]["corpus.jsonl"]) == 64
    pairs = load_pairs(out / "pairs.jsonl")
    assert pairs[0]["positive"] == "Title First"
    assert pairs[0]["positives"] == ["Second"]
    assert json.loads((out / "manifest.json").read_text()) == manifest


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_judgments_fail_before_writing_dataset(tmp_path, score):
    source = tmp_path / "source"
    (source / "qrels").mkdir(parents=True)
    (source / "corpus.jsonl").write_text('{"_id":"d","text":"Document"}\n')
    (source / "queries.jsonl").write_text('{"_id":"q","text":"Question"}\n')
    (source / "qrels/test.tsv").write_text(f'query-id\tcorpus-id\tscore\nq\td\t{score}\n')
    with pytest.raises(ValueError, match="finite"):
        prepare(source, "test", tmp_path / "output")
    assert not (tmp_path / "output").exists()
