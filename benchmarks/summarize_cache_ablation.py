"""Validate pairing and summarize the fixed SciFact cache experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.cache_ablation_data import paired_bootstrap
from retrieval import file_digest


ARMS = ("reference", "plain", "autodiff", "float32")
SEEDS = (41, 42, 43)


def metric_value(row, metric):
    return row["recall_at"]["10"] if metric == "recall_at_10" else row[metric]


def validate_pairing(baseline, runs):
    for (arm, seed), report in runs.items():
        if report["arm"] != arm or report["seed"] != seed:
            raise ValueError("Mismatched run identity")
        for field in ("protocol", "manifest_sha256", "source_sha256", "versions"):
            if report[field] != baseline[field]:
                raise ValueError(f"Mismatched {field}")
        for split in ("test", "validation"):
            if report["evaluation"][split]["query_ids"] != baseline["evaluation"][split]["query_ids"]:
                raise ValueError("Mismatched query order")
    for seed in SEEDS:
        target = runs[("reference", seed)]["training"]
        for arm in ARMS:
            actual = runs[(arm, seed)]["training"]
            for key in ("initial_adapter_sha256", "order_sha256", "steps"):
                if actual[key] != target[key]:
                    raise ValueError(f"Mismatched paired {key}: {arm}/{seed}")
            if [(r["rng_before"], r["rng_after"]) for r in actual["history"]] != [
                    (r["rng_before"], r["rng_after"]) for r in target["history"]]:
                raise ValueError(f"Mismatched paired RNG: {arm}/{seed}")


def adapter_array(path):
    from safetensors.numpy import load_file
    values = load_file(str(path))
    return np.concatenate([values[name].ravel().astype(np.float64) for name in sorted(values)])


def summarize(root, manifest):
    baseline = json.loads((root / "baseline-41/result.json").read_text())
    reports = {(arm, seed): json.loads((root / f"{arm}-{seed}/result.json").read_text())
               for arm in ARMS for seed in SEEDS}
    validate_pairing(baseline, reports)
    if baseline["manifest_sha256"] != file_digest(manifest):
        raise ValueError("Manifest differs from the experiment")
    data = json.loads(manifest.read_text())
    metrics = ("ndcg_at_10", "mrr_at_10", "recall_at_10")
    matrices = {arm: {metric: np.array([
        [metric_value(r, metric) for r in reports[(arm, seed)]["evaluation"]["test"]["per_query"]]
        for seed in SEEDS]) for metric in metrics} for arm in ARMS}
    summary = {"protocol": baseline["protocol"], "versions": baseline["versions"],
               "source_sha256": baseline["source_sha256"],
               "manifest": {k: v for k, v in data.items() if k != "test_query_groups"},
               "baseline": {split: {metric: metric_value(baseline["evaluation"][split], metric)
                                     for metric in metrics} for split in ("test", "validation")},
               "runs": [], "comparisons": {}}
    for arm in ARMS:
        for seed in SEEDS:
            report = reports[(arm, seed)]
            training = report["training"]
            summary["runs"].append({"arm": arm, "seed": seed,
                "test": {metric: metric_value(report["evaluation"]["test"], metric) for metric in metrics},
                "validation": {metric: metric_value(report["evaluation"]["validation"], metric) for metric in metrics},
                "training": {k: v for k, v in training.items() if k != "history"},
                "result_sha256": file_digest(root / f"{arm}-{seed}/result.json"),
                "adapter_sha256": file_digest(root / f"{arm}-{seed}/adapters.safetensors")})
    for a, b in (("autodiff", "plain"), ("autodiff", "reference"), ("plain", "reference"), ("float32", "plain")):
        summary["comparisons"][f"{a}_minus_{b}"] = {
            metric: paired_bootstrap(matrices[a][metric] - matrices[b][metric], data["test_query_groups"])
            for metric in metrics}
    summary["adapter_trajectory_distances"] = []
    for seed in SEEDS:
        initial = adapter_array(root / f"reference-{seed}/initial_adapters.safetensors")
        for step in (1, 21, 41):
            ref = adapter_array(root / f"reference-{seed}/adapters-step-{step}.safetensors")
            update_norm = np.linalg.norm(ref-initial)
            for arm in ("plain", "autodiff", "float32"):
                actual = adapter_array(root / f"{arm}-{seed}/adapters-step-{step}.safetensors")
                distance = np.linalg.norm(actual-ref)
                summary["adapter_trajectory_distances"].append({
                    "seed": seed, "arm": arm, "step": step,
                    "distance_to_reference": float(distance),
                    "reference_update_norm": float(update_norm),
                    "distance_relative_to_reference_update": float(distance / max(update_norm, 1e-30))})
    group_of = {i: group_id for group_id, group in enumerate(data["test_query_groups"]) for i in group}
    per_query = []
    for i, qid in enumerate(baseline["evaluation"]["test"]["query_ids"]):
        per_query.append({"query_id": qid, "group": group_of[i],
                          "baseline": {metric: metric_value(baseline["evaluation"]["test"]["per_query"][i], metric)
                                       for metric in metrics},
                          "seeds": list(SEEDS),
                          "arms": {arm: {metric: matrices[arm][metric][:, i].tolist() for metric in metrics}
                                   for arm in ARMS}})
    return summary, per_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-query-output", required=True)
    args = parser.parse_args()
    out, detail = Path(args.output), Path(args.per_query_output)
    if out.exists() or detail.exists() or out.is_symlink() or detail.is_symlink():
        parser.error("Choose fresh result paths")
    summary, per_query = summarize(Path(args.runs_dir), Path(args.manifest))
    with out.open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    with detail.open("x") as f:
        for row in per_query:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
