#!/usr/bin/env python3
"""Validate common scMultiBench input files before running heavy workflows."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class H5Summary:
    path: Path
    key: str
    shape: tuple[int, ...]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        fail(f"file does not exist: {resolved}")
    if not resolved.is_file():
        fail(f"path is not a file: {resolved}")
    return resolved


def import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        fail("h5py is required for H5 validation. Install envs/evaluation-python.yml first.")
    return h5py


def h5_shape(path: str | Path, key: str = "data") -> H5Summary:
    h5py = import_h5py()
    resolved = require_file(path)
    with h5py.File(resolved, "r") as handle:
        if key not in handle:
            available = sorted(handle.keys())
            fail(f"{resolved} is missing H5 dataset '{key}'. Top-level keys: {available}")
        dataset = handle[key]
        shape = tuple(int(dim) for dim in dataset.shape)
    return H5Summary(resolved, key, shape)


def _looks_like_header(row: list[str]) -> bool:
    lowered = [cell.strip().lower() for cell in row]
    return "x" in lowered or "celltype" in lowered or "label" in lowered


def read_labels(path: str | Path, column: str = "auto") -> list[str]:
    resolved = require_file(path)
    with resolved.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        fail(f"label file is empty: {resolved}")

    header = rows[0]
    data_rows = rows[1:] if _looks_like_header(header) else rows

    if column != "auto":
        if _looks_like_header(header):
            try:
                index = header.index(column)
            except ValueError:
                fail(f"label column '{column}' not found in {resolved}; columns are {header}")
        else:
            try:
                index = int(column)
            except ValueError:
                fail(f"{resolved} has no header; use a numeric --label-column index")
    elif _looks_like_header(header):
        lowered = [cell.strip().lower() for cell in header]
        if "x" in lowered:
            index = lowered.index("x")
        elif "celltype" in lowered:
            index = lowered.index("celltype")
        elif "label" in lowered:
            index = lowered.index("label")
        else:
            index = len(header) - 1
    else:
        index = 1 if len(header) > 1 else 0

    labels: list[str] = []
    for row_number, row in enumerate(data_rows, start=2 if data_rows is not rows else 1):
        if index >= len(row):
            fail(f"row {row_number} in {resolved} has no column index {index}")
        value = row[index].strip()
        if value:
            labels.append(value)
    if not labels:
        fail(f"no non-empty labels found in {resolved}")
    return labels


def count_labels(paths: Iterable[str | Path], column: str = "auto") -> int:
    return sum(len(read_labels(path, column)) for path in paths)


def display_shape(summary: H5Summary) -> str:
    shape = " x ".join(str(dim) for dim in summary.shape)
    return f"{summary.path} [{summary.key}: {shape}]"


def validate_rows(name: str, shape: tuple[int, ...], expected_rows: int) -> None:
    if not shape:
        fail(f"{name} is scalar; expected a matrix-like dataset")
    rows = shape[0]
    transposed_rows = shape[1] if len(shape) > 1 else None
    if rows == expected_rows:
        return
    if transposed_rows == expected_rows:
        print(f"NOTE: {name} appears transposed; legacy scripts may transpose it automatically.")
        return
    fail(f"{name} has {rows} rows, but labels contain {expected_rows} rows")


def validate_labels(args: argparse.Namespace) -> None:
    total = count_labels(args.labels, args.label_column)
    if args.expected_rows is not None and total != args.expected_rows:
        fail(f"labels contain {total} rows, expected {args.expected_rows}")
    print(f"OK labels: {total} rows across {len(args.labels)} file(s)")


def validate_classification(args: argparse.Namespace) -> None:
    reference = h5_shape(args.reference, args.data_key)
    query = h5_shape(args.query, args.data_key)
    reference_rows = count_labels([args.reference_labels], args.label_column)
    query_rows = count_labels([args.query_labels], args.label_column)
    validate_rows("reference", reference.shape, reference_rows)
    validate_rows("query", query.shape, query_rows)
    print(f"OK reference: {display_shape(reference)} with {reference_rows} labels")
    print(f"OK query: {display_shape(query)} with {query_rows} labels")


def validate_scib_embedding(args: argparse.Namespace) -> None:
    embedding = h5_shape(args.embedding, args.data_key)
    label_rows = count_labels(args.labels, args.label_column)
    validate_rows("embedding", embedding.shape, label_rows)
    cluster = h5_shape(args.cluster, args.cluster_key)
    batch_cluster = h5_shape(args.batch_cluster, args.cluster_key)
    validate_rows("cluster", cluster.shape, label_rows)
    validate_rows("batch cluster", batch_cluster.shape, label_rows)
    print(f"OK embedding: {display_shape(embedding)} with {label_rows} labels")
    print(f"OK cluster: {display_shape(cluster)}")
    print(f"OK batch cluster: {display_shape(batch_cluster)}")


def validate_scib_graph(args: argparse.Namespace) -> None:
    indices = h5_shape(args.knn_indices, args.data_key)
    dists = h5_shape(args.knn_dists, args.data_key)
    label_rows = count_labels(args.labels, args.label_column)
    validate_rows("knn indices", indices.shape, label_rows)
    validate_rows("knn dists", dists.shape, label_rows)
    cluster = h5_shape(args.cluster, args.cluster_key)
    batch_cluster = h5_shape(args.batch_cluster, args.cluster_key)
    validate_rows("cluster", cluster.shape, label_rows)
    validate_rows("batch cluster", batch_cluster.shape, label_rows)
    print(f"OK knn indices: {display_shape(indices)} with {label_rows} labels")
    print(f"OK knn dists: {display_shape(dists)}")
    print(f"OK cluster: {display_shape(cluster)}")
    print(f"OK batch cluster: {display_shape(batch_cluster)}")


def validate_imputation(args: argparse.Namespace) -> None:
    real = h5_shape(args.real, args.data_key)
    imputed = h5_shape(args.imputed, args.data_key)
    if real.shape != imputed.shape:
        fail(f"real and imputed matrices have different shapes: {real.shape} vs {imputed.shape}")
    if args.labels:
        label_rows = count_labels([args.labels], args.label_column)
        validate_rows("real", real.shape, label_rows)
        print(f"OK imputation labels: {label_rows} rows")
    print(f"OK real: {display_shape(real)}")
    print(f"OK imputed: {display_shape(imputed)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    labels = subparsers.add_parser("labels", help="Validate label CSV files only")
    labels.add_argument("--labels", nargs="+", required=True)
    labels.add_argument("--label-column", default="auto")
    labels.add_argument("--expected-rows", type=int)
    labels.set_defaults(func=validate_labels)

    classification = subparsers.add_parser("classification", help="Validate classification H5 and label files")
    classification.add_argument("--reference", required=True)
    classification.add_argument("--query", required=True)
    classification.add_argument("--reference-labels", required=True)
    classification.add_argument("--query-labels", required=True)
    classification.add_argument("--data-key", default="data")
    classification.add_argument("--label-column", default="auto")
    classification.set_defaults(func=validate_classification)

    scib_embedding = subparsers.add_parser("scib-embedding", help="Validate embedding-based scIB metric inputs")
    scib_embedding.add_argument("--embedding", required=True)
    scib_embedding.add_argument("--labels", nargs="+", required=True)
    scib_embedding.add_argument("--cluster", required=True)
    scib_embedding.add_argument("--batch-cluster", required=True)
    scib_embedding.add_argument("--data-key", default="data")
    scib_embedding.add_argument("--cluster-key", default="obs/cluster_leiden")
    scib_embedding.add_argument("--label-column", default="auto")
    scib_embedding.set_defaults(func=validate_scib_embedding)

    scib_graph = subparsers.add_parser("scib-graph", help="Validate graph-based scIB metric inputs")
    scib_graph.add_argument("--knn-indices", required=True)
    scib_graph.add_argument("--knn-dists", required=True)
    scib_graph.add_argument("--labels", nargs="+", required=True)
    scib_graph.add_argument("--cluster", required=True)
    scib_graph.add_argument("--batch-cluster", required=True)
    scib_graph.add_argument("--data-key", default="data")
    scib_graph.add_argument("--cluster-key", default="obs/cluster_leiden")
    scib_graph.add_argument("--label-column", default="auto")
    scib_graph.set_defaults(func=validate_scib_graph)

    imputation = subparsers.add_parser("imputation", help="Validate imputation real/imputed H5 files")
    imputation.add_argument("--real", required=True)
    imputation.add_argument("--imputed", required=True)
    imputation.add_argument("--labels")
    imputation.add_argument("--data-key", default="data")
    imputation.add_argument("--label-column", default="auto")
    imputation.set_defaults(func=validate_imputation)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
