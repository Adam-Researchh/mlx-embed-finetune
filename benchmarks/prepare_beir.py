#!/usr/bin/env python3
"""Convert a local BEIR dataset to this project's text-based retrieval format.

Run with: python -m benchmarks.prepare_beir --data-dir /path/to/scifact \
    --split test --output-dir outputs/scifact-test
Judgments with score > 0 are treated as binary relevance, as in SciFact.
"""

import argparse
import csv
import json
from pathlib import Path

from retrieval import file_digest, read_jsonl


def prepare(data_dir, split, output_dir):
    root, out = Path(data_dir), Path(output_dir)
    corpus_path, queries_path = root / "corpus.jsonl", root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    corpus, queries = {}, {}
    for _, row in read_jsonl(corpus_path):
        doc_id = str(row["_id"])
        if doc_id in corpus:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        corpus[doc_id] = " ".join(part.strip() for part in
                                  (row.get("title", ""), row["text"]) if part.strip())
    for _, row in read_jsonl(queries_path):
        query_id = str(row["_id"])
        if query_id in queries:
            raise ValueError(f"Duplicate query ID: {query_id}")
        queries[query_id] = row["text"]
    relevant = {}
    with qrels_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            query_id, doc_id = row["query-id"], row["corpus-id"]
            if query_id not in queries or doc_id not in corpus:
                raise ValueError(f"Unresolved judgment: {query_id}/{doc_id}")
            if float(row["score"]) > 0:
                relevant.setdefault(query_id, []).append(doc_id)
    if not relevant:
        raise ValueError("No positive judgments in this split")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("x", encoding="utf-8") as stream:
        for doc_id, text in corpus.items():
            stream.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    with (out / "pairs.jsonl").open("x", encoding="utf-8") as stream:
        for query_id, doc_ids in relevant.items():
            positives = list(dict.fromkeys(corpus[doc_id] for doc_id in doc_ids))
            stream.write(json.dumps({"query_id": query_id, "query": queries[query_id],
                                     "positive": positives[0], "positives": positives[1:]},
                                    ensure_ascii=False) + "\n")
    manifest = {"split": split, "relevance": "binary (score > 0)",
                "queries": len(relevant), "documents": len(corpus),
                "source_sha256": {str(p.relative_to(root)): file_digest(p)
                                  for p in (corpus_path, queries_path, qrels_path)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.split, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
