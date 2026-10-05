"""Rebuild ``aggregate_data/`` from the released raw-data tables.

Recomputes every metric from the raw counters and rejection-sampling
trajectories on Hugging Face, orders models as in the paper, validates the
result, and writes the 108 aggregate CSV files.
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

DATASETS = (
    "aime",
    "csqa",
    "gpqa",
    "gsm",
    "matmul",
    "mmlu_social_sciences",
    "mmlu_stem",
    "sat",
    "sudoku",
)

MODELS_37 = (
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)

MODELS_12 = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
)


def parse_args() -> argparse.Namespace:
    """Parse reconstruction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data",
        default="Jacklu0831/llm-verification-raw",
        help="Hugging Face dataset ID or downloaded dataset repository path.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("aggregate_data"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing aggregate CSVs after a complete rebuild succeeds.",
    )
    return parser.parse_args()


def model_display_name(model_id: str) -> str:
    """Return the display label used by the committed aggregate CSVs."""
    if model_id.startswith("Qwen/Qwen3-"):
        return model_id.removeprefix("Qwen/")
    if model_id.startswith("Qwen/Qwen2.5-"):
        name = model_id.removeprefix("Qwen/")
        return (
            name.removesuffix("-Instruct")
            if name.endswith("-Instruct")
            else f"{name}-Base"
        )
    if model_id.startswith("meta-llama/Llama-"):
        size = model_id.removesuffix("-Instruct").rsplit("-", maxsplit=1)[-1]
        suffix = "" if model_id.endswith("-Instruct") else "-Base"
        return f"Llama3-{size}{suffix}"
    if model_id.startswith("deepseek-ai/DeepSeek-R1-Distill-Qwen-"):
        size = model_id.rsplit("-", maxsplit=1)[-1]
        return f"DeepSeek-{size}"
    raise ValueError(f"Unknown paper model ID: {model_id}")


def safe_divide(numerator: float, denominator: float, default: float) -> float:
    """Return numerator / denominator, or `default` when the denominator is zero."""
    if denominator < 0:
        raise ValueError(f"Negative denominator: {denominator}")
    return numerator / denominator if denominator > 0 else default


def recompute_run_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    """Recompute one run's metrics from its raw counters."""
    solver_total = int(row["solver_total"])
    solver_bad = int(row["solver_bad_count"])
    solver_correct = int(row["solver_correct_count"])
    tp = int(row["true_positive"])
    tn = int(row["true_negative"])
    fp = int(row["false_positive"])
    fn = int(row["false_negative"])

    counters = {
        "solver_total": solver_total,
        "solver_bad_count": solver_bad,
        "solver_correct_count": solver_correct,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "verifier_bad_count": int(row["verifier_bad_count"]),
    }
    negative = {name: value for name, value in counters.items() if value < 0}
    if negative:
        raise ValueError(f"Negative raw counters: {negative}")

    solver_parseable = solver_total - solver_bad
    if solver_parseable < 0 or solver_correct > solver_parseable:
        raise ValueError(f"Invalid solver counters: {counters}")
    verifier_parseable = tp + tn + fp + fn
    if verifier_parseable + counters["verifier_bad_count"] != solver_parseable:
        raise ValueError(f"Inconsistent solver/verifier counters: {counters}")

    solver_accuracy = safe_divide(solver_correct, solver_parseable, 0.0)
    verifier_accuracy = safe_divide(tp + tn, tp + tn + fp + fn, 0.0)
    precision = safe_divide(tp, tp + fp, 0.0)
    recall = safe_divide(tp, tp + fn, 0.0)
    f1 = safe_divide(2.0 * precision * recall, precision + recall, 0.0)
    false_positive_rate = safe_divide(fp, fp + tn, 1.0)
    false_negative_rate = safe_divide(fn, tp + fn, 1.0)
    acceptance_probability = (
        solver_accuracy * recall + (1.0 - solver_accuracy) * false_positive_rate
    )
    accepted_accuracy = safe_divide(
        solver_accuracy * recall,
        acceptance_probability,
        0.0,
    )
    return {
        "solver_accuracy": solver_accuracy,
        "solver_filter_rate": safe_divide(solver_bad, solver_total, 0.0),
        "verifier_accuracy": verifier_accuracy,
        "verifier_precision": precision,
        "verifier_f1": f1,
        "verifier_fpr": false_positive_rate,
        "verifier_fnr": false_negative_rate,
        "verifier_gain": accepted_accuracy - solver_accuracy,
    }


def assert_close(name: str, observed: float, expected: float) -> None:
    """Fail if a released derived value disagrees with its raw counters."""
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{name}: observed={observed}, expected={expected}")


def index_unique(
    rows: Iterable[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    """Index rows by a composite key and reject duplicates."""
    indexed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in indexed:
            raise AssertionError(f"Duplicate row key: {key}")
        indexed[key] = row
    return indexed


def write_csv(path: Path, rows: Iterable[Sequence[str]]) -> None:
    """Write one aggregate CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def write_vector_metrics(
    output_dir: Path,
    metrics: Mapping[tuple[str, str, str], float],
) -> None:
    """Write solver accuracy and filter-rate vectors."""
    for dataset in DATASETS:
        for metric_name, value_header in (
            ("solver_accuracy", "accuracy"),
            ("solver_filter_rate", "filter_rate"),
        ):
            rows: list[Sequence[str]] = [("model", value_header)]
            rows.extend(
                (
                    model_display_name(solver_model),
                    f"{metrics[(dataset, solver_model, metric_name)]:.6f}",
                )
                for solver_model in MODELS_37
            )
            write_csv(output_dir / metric_name / f"{dataset}.csv", rows)


def write_matrix(
    path: Path,
    models: Sequence[str],
    values: Mapping[tuple[str, str], float],
) -> None:
    """Write one solver-by-verifier matrix in fixed paper order."""
    rows: list[Sequence[str]] = [
        ("solver", *(model_display_name(model) for model in models))
    ]
    rows.extend(
        (
            model_display_name(solver_model),
            *(
                f"{values[(solver_model, verifier_model)]:.6f}"
                for verifier_model in models
            ),
        )
        for solver_model in models
    )
    write_csv(path, rows)


def rebuild_ordinary(run_metrics: Dataset, output_dir: Path) -> None:
    """Recompute the per-model vectors and solver-by-verifier matrices from raw counts."""
    expected_rows = len(DATASETS) * len(MODELS_37) ** 2
    if len(run_metrics) != expected_rows:
        raise AssertionError(
            f"run_metrics has {len(run_metrics)} rows; expected {expected_rows}"
        )

    indexed = index_unique(
        run_metrics,
        ("dataset", "solver_model", "verifier_model"),
    )
    solver_values: dict[tuple[str, str, str], float] = {}
    matrix_metrics = (
        ("verifier_accuracy", "verifier_accuracy"),
        ("verifier_precision", "precision"),
        ("verifier_f1", "f1"),
        ("verifier_fpr", "false_positive_rate"),
        ("verifier_fnr", "false_negative_rate"),
        ("verifier_gain", "verifier_gain"),
    )

    for dataset in DATASETS:
        for solver_model in MODELS_37:
            solver_baseline: tuple[float, float] | None = None
            for verifier_model in MODELS_37:
                row = indexed[(dataset, solver_model, verifier_model)]
                computed = recompute_run_metrics(row)
                for released_name, source_name in matrix_metrics:
                    assert_close(
                        released_name,
                        float(row[source_name]),
                        computed[released_name],
                    )
                current = (
                    computed["solver_accuracy"],
                    computed["solver_filter_rate"],
                )
                if solver_baseline is None:
                    solver_baseline = current
                elif current != solver_baseline:
                    raise AssertionError(
                        f"Solver counters vary across verifiers: {dataset}/{solver_model}"
                    )
            assert solver_baseline is not None
            solver_values[(dataset, solver_model, "solver_accuracy")] = solver_baseline[
                0
            ]
            solver_values[(dataset, solver_model, "solver_filter_rate")] = (
                solver_baseline[1]
            )

    write_vector_metrics(output_dir, solver_values)
    for dataset in DATASETS:
        for metric_name, _ in matrix_metrics:
            values = {
                (solver_model, verifier_model): recompute_run_metrics(
                    indexed[(dataset, solver_model, verifier_model)]
                )[metric_name]
                for solver_model in MODELS_37
                for verifier_model in MODELS_37
            }
            write_matrix(
                output_dir / metric_name / f"{dataset}.csv",
                MODELS_37,
                values,
            )


def rebuild_rejection_sampling(
    rejection_sampling: Dataset,
    output_dir: Path,
) -> None:
    """Recompute K=3 and K=10 empirical gains from attempt trajectories."""
    expected_rows = len(DATASETS) * len(MODELS_12) ** 2 * 10
    if len(rejection_sampling) != expected_rows:
        raise AssertionError(
            f"rejection_sampling has {len(rejection_sampling)} rows; expected {expected_rows}"
        )
    indexed = index_unique(
        rejection_sampling,
        ("dataset", "solver_model", "verifier_model", "attempt"),
    )
    for dataset in DATASETS:
        for attempt, output_name in (
            (2, "empirical_gain_k3"),
            (9, "empirical_gain_k10"),
        ):
            values: dict[tuple[str, str], float] = {}
            for solver_model in MODELS_12:
                for verifier_model in MODELS_12:
                    first = indexed[(dataset, solver_model, verifier_model, 0)]
                    current = indexed[(dataset, solver_model, verifier_model, attempt)]
                    gain = float(current["solver_accuracy"]) - float(
                        first["solver_accuracy"]
                    )
                    assert_close(
                        "accuracy_gain_from_first_candidate",
                        float(current["accuracy_gain_from_first_candidate"]),
                        gain,
                    )
                    values[(solver_model, verifier_model)] = gain
            write_matrix(
                output_dir / output_name / f"{dataset}.csv",
                MODELS_12,
                values,
            )


def rebuild_similarity(similarity: Dataset, output_dir: Path) -> None:
    """Pivot the two released similarity measures into 12-by-12 matrices."""
    expected_rows = len(DATASETS) * len(MODELS_12) ** 2 * 2
    if len(similarity) != expected_rows:
        raise AssertionError(
            f"similarity has {len(similarity)} rows; expected {expected_rows}"
        )
    indexed = index_unique(
        similarity,
        ("dataset", "similarity_measure", "solver_model", "verifier_model"),
    )
    measure_names = {
        "embedding_cosine": "similarity",
        "generation_likelihood": "similarity_likelihood",
    }
    for dataset in DATASETS:
        for measure, output_name in measure_names.items():
            values = {
                (solver_model, verifier_model): float(
                    indexed[(dataset, measure, solver_model, verifier_model)][
                        "similarity"
                    ]
                )
                for solver_model in MODELS_12
                for verifier_model in MODELS_12
            }
            write_matrix(
                output_dir / output_name / f"{dataset}.csv",
                MODELS_12,
                values,
            )


def expected_csv_paths() -> set[Path]:
    """Return the exact aggregate CSV paths produced by this script."""
    metric_names = {
        "solver_accuracy",
        "solver_filter_rate",
        "verifier_accuracy",
        "verifier_precision",
        "verifier_f1",
        "verifier_fpr",
        "verifier_fnr",
        "verifier_gain",
        "empirical_gain_k3",
        "empirical_gain_k10",
        "similarity",
        "similarity_likelihood",
    }
    return {
        Path(metric_name) / f"{dataset}.csv"
        for metric_name in metric_names
        for dataset in DATASETS
    }


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Reject accidental writes outside the known aggregate tree."""
    if not output_dir.exists():
        return
    existing_files = {
        path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()
    }
    unexpected = existing_files - expected_csv_paths()
    if unexpected:
        raise FileExistsError(f"Unexpected files in {output_dir}: {sorted(unexpected)}")
    if existing_files and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains aggregate CSVs; pass --overwrite"
        )


def install_rebuilt_csvs(staging_dir: Path, output_dir: Path) -> int:
    """Install a validated rebuild with atomic replacement of each CSV."""
    expected = expected_csv_paths()
    observed = {path.relative_to(staging_dir) for path in staging_dir.rglob("*.csv")}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise AssertionError(f"CSV path mismatch: missing={missing}, extra={extra}")

    changed = 0
    for relative_path in sorted(expected):
        source = staging_dir / relative_path
        destination = output_dir / relative_path
        if not destination.exists() or source.read_bytes() != destination.read_bytes():
            changed += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    return changed


def load_config(args: argparse.Namespace, config_name: str) -> Dataset:
    """Load one reconstruction configuration from the release dataset."""
    return load_dataset(
        args.raw_data,
        config_name,
        split="data",
        revision=args.revision,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
    )


def main() -> None:
    """Load the raw data and rebuild aggregate_data/."""
    args = parse_args()
    validate_output_dir(args.output_dir, args.overwrite)

    run_metrics = load_config(args, "run_metrics")
    rejection_sampling = load_config(args, "rejection_sampling")
    similarity = load_config(args, "similarity")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}-",
        dir=args.output_dir.parent,
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        rebuild_ordinary(run_metrics, staging_dir)
        rebuild_rejection_sampling(rejection_sampling, staging_dir)
        rebuild_similarity(similarity, staging_dir)
        changed = install_rebuilt_csvs(staging_dir, args.output_dir)
    print(f"Rebuilt 108 aggregate CSVs in {args.output_dir} ({changed} changed).")


if __name__ == "__main__":
    main()
