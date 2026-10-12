"""Pure NumPy preparation and paired analysis for the SciFact cache pilot."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np

from benchmarks.prepare_beir import prepare as prepare_beir
from retrieval import file_digest, known_positives, load_corpus, load_pairs, positive_texts


def positive_groups(rows):
    """Connected row groups sharing exact query or positive-document text."""
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in [("q", row["query"]), *[("d", d) for d in positive_texts(row)]]:
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    return sorted(groups.values(), key=lambda group: group[0])


def split_rows(rows, test_rows, seed=20261012, validation_fraction=.2):
    test_queries = {row["query"] for row in test_rows}
    removed = [row for row in rows if row["query"] in test_queries]
    eligible = [row for row in rows if row["query"] not in test_queries]
    groups = positive_groups(eligible)
    if len(groups) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least two positive groups and a fraction inside (0, 1)")
    random.Random(seed).shuffle(groups)
    val_indices = set()
    for group in groups[:-1]:
        if len(val_indices) >= len(eligible) * validation_fraction:
            break
        val_indices.update(group)
    training = [row for i, row in enumerate(eligible) if i not in val_indices]
    validation = [row for i, row in enumerate(eligible) if i in val_indices]
    return training, validation, removed


def bm25_negatives(rows, corpus, count=2):
    """Fixed lexical mining; unjudged documents are assumed negative."""
    if count < 1 or not corpus:
        raise ValueError("Need a positive negative count and a corpus")
    tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    postings = defaultdict(list)
    lengths = np.zeros(len(corpus), np.float64)
    for i, text in enumerate(corpus):
        terms = Counter(tokenize(text))
        lengths[i] = sum(terms.values())
        for term, tf in terms.items():
            postings[term].append((i, tf))
    norm = 1.5 * (.25 + .75 * lengths / max(lengths.mean(), 1))
    table = {}
    for term, entries in postings.items():
        ids = np.array([i for i, _ in entries], dtype=np.int64)
        tf = np.array([tf for _, tf in entries], dtype=np.float64)
        idf = np.log1p((len(corpus) - len(ids) + .5) / (len(ids) + .5))
        table[term] = ids, idf * tf * 2.5 / (tf + norm[ids])
    relevant = known_positives(rows)
    output = []
    for row in rows:
        scores = np.zeros(len(corpus), np.float64)
        for term in sorted(set(tokenize(row["query"]))):
            if term in table:
                ids, values = table[term]
                scores[ids] += values
        order = np.lexsort((np.arange(len(corpus)), -scores))
        negatives = []
        for i in order:
            text = corpus[i]
            if text not in relevant[row["query"]] and text not in negatives:
                negatives.append(text)
            if len(negatives) == count:
                break
        if len(negatives) < count:
            raise ValueError("Insufficient nonpositive documents")
        output.append({**row, "negatives": negatives})
    return output


def paired_bootstrap(differences, groups, draws=10000, seed=20261012):
    """Resample paired seed and document-group units; preserve query weights.

    differences is [seed, query]. The same sampled groups apply to all sampled
    seeds in a replicate. With few seeds these intervals are exploratory.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Need finite seed-by-query differences")
    if sorted(i for group in groups for i in group) != list(range(values.shape[1])):
        raise ValueError("Groups must partition query indices exactly once")
    if any(not group for group in groups) or draws < 1:
        raise ValueError("Nonempty groups and positive draws required")
    totals = np.stack([values[:, group].sum(axis=1) for group in groups], axis=1)
    sizes = np.array([len(group) for group in groups])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        seeds = rng.integers(values.shape[0], size=values.shape[0])
        sampled = rng.integers(len(groups), size=len(groups))
        means.append(float(totals[seeds][:, sampled].sum() / (len(seeds) * sizes[sampled].sum())))
    return {"mean_delta": float(values.mean()), "per_seed_delta": values.mean(axis=1).tolist(),
            "exploratory_95_interval": np.quantile(means, [.025, .975]).tolist(),
            "draws": draws, "seed": seed, "query_groups": len(groups)}


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a new output directory")
    source = Path(args.data_dir)
    prepare_beir(source, "train", out / "official-train")
    prepare_beir(source, "test", out / "official-test")
    official_train = load_pairs(out / "official-train/pairs.jsonl")
    test = load_pairs(out / "official-test/pairs.jsonl")
    corpus = load_corpus(out / "official-train/corpus.jsonl")
    training, validation, removed = split_rows(official_train, test)
    mined = bm25_negatives(training, corpus)
    write_rows(out / "train.jsonl", mined)
    write_rows(out / "validation.jsonl", validation)
    write_rows(out / "test.jsonl", test)
    corpus_rows = [json.loads(line) for line in (out / "official-train/corpus.jsonl").read_text().splitlines()]
    write_rows(out / "corpus.jsonl", corpus_rows)
    files = ["train.jsonl", "validation.jsonl", "test.jsonl", "corpus.jsonl"]
    manifest = {"official_train_queries": len(official_train), "train_queries": len(training),
                "validation_queries": len(validation), "test_queries": len(test), "corpus_documents": len(corpus),
                "excluded_train_query_ids": [r["query_id"] for r in removed],
                "split_seed": 20261012, "validation_fraction_target": .2,
                "split_method": "connected components of exact query and positive-document text",
                "negative_mining": "BM25 k1=1.5 b=.75; two unjudged negatives; same fixed candidates for every arm",
                "file_sha256": {name: file_digest(out / name) for name in files},
                "test_query_groups": positive_groups(test),
                "source_sha256": {str(p.relative_to(source)): file_digest(p) for p in
                                  [source / "corpus.jsonl", source / "queries.jsonl", source / "qrels/train.tsv", source / "qrels/test.tsv"]}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "test_query_groups"}, indent=2))


if __name__ == "__main__":
    main()
