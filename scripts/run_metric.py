#!/usr/bin/env python3
"""Dispatch common scMultiBench metric workflows through stable commands."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], dry_run: bool) -> int:
    print(quote_command(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated command without running it")


def scib_embedding(args: argparse.Namespace) -> int:
    script = repo_root() / "evaluation_pipelines" / "scib_metrics" / "scib_metrics.py"
    command = [
        sys.executable,
        str(script),
        "--data_path",
        args.embedding,
        "--cty_path",
        *args.labels,
        "--cluster_path",
        args.cluster,
        "--batch_cluster_path",
        args.batch_cluster,
        "--save_path",
        args.out,
    ]
    return run_command(command, args.dry_run)


def scib_graph(args: argparse.Namespace) -> int:
    script = repo_root() / "evaluation_pipelines" / "scib_metrics" / "scib_metrics_graph.py"
    command = [
        sys.executable,
        str(script),
        "--knn_indices",
        args.knn_indices,
        "--knn_dists",
        args.knn_dists,
        "--cty_path",
        *args.labels,
        "--cluster_path",
        args.cluster,
        "--batch_cluster_path",
        args.batch_cluster,
        "--save_path",
        args.out,
    ]
    return run_command(command, args.dry_run)


def sinfonia_clustering(args: argparse.Namespace) -> int:
    script_name = "SINFONIA_batch.py" if args.batch else "SINFONIA.py"
    script = repo_root() / "evaluation_pipelines" / "clustering" / script_name
    command = [sys.executable, str(script), "--path1", args.embedding, "--save_path", args.out]
    if args.batch:
        if args.num is None:
            raise SystemExit("--num is required when using --batch")
        command.extend(["--num", str(args.num)])
    else:
        if not args.labels:
            raise SystemExit("--labels is required unless --batch is set")
        command.extend(["--cty_path", *args.labels])
    return run_command(command, args.dry_run)


def mlp_classification(args: argparse.Namespace) -> int:
    script_dir = repo_root() / "evaluation_pipelines" / "classification" / "MLP_classification"
    script = script_dir / "main.py"
    command = [
        sys.executable,
        str(script),
        "--data_path1",
        args.reference,
        "--data_path2",
        *args.query,
        "--cty_path1",
        args.reference_labels,
        "--cty_path2",
        *args.query_labels,
        "--save_path",
        args.out,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]
    print(f"cd {shlex.quote(str(script_dir))}")
    if args.dry_run:
        print(quote_command(command))
        return 0
    return subprocess.run(command, check=False, cwd=script_dir).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scib_emb = subparsers.add_parser("scib-embedding", help="Run embedding-based scIB-style metrics")
    scib_emb.add_argument("--embedding", required=True, help="H5 embedding file with dataset `data`")
    scib_emb.add_argument("--labels", nargs="+", required=True, help="One or more cell-type CSV files")
    scib_emb.add_argument("--cluster", required=True, help="H5 clustering file with obs/cluster_leiden")
    scib_emb.add_argument("--batch-cluster", required=True, help="H5 batch-clustering file with obs/cluster_leiden")
    scib_emb.add_argument("--out", required=True, help="Output directory")
    common_options(scib_emb)
    scib_emb.set_defaults(func=scib_embedding)

    scib_g = subparsers.add_parser("scib-graph", help="Run graph-based scIB-style metrics")
    scib_g.add_argument("--knn-indices", required=True, help="H5 KNN index file with dataset `data`")
    scib_g.add_argument("--knn-dists", required=True, help="H5 KNN distance file with dataset `data`")
    scib_g.add_argument("--labels", nargs="+", required=True, help="One or more cell-type CSV files")
    scib_g.add_argument("--cluster", required=True, help="H5 clustering file with obs/cluster_leiden")
    scib_g.add_argument("--batch-cluster", required=True, help="H5 batch-clustering file with obs/cluster_leiden")
    scib_g.add_argument("--out", required=True, help="Output directory")
    common_options(scib_g)
    scib_g.set_defaults(func=scib_graph)

    sinfonia = subparsers.add_parser("sinfonia-clustering", help="Run SINFONIA clustering on an embedding file")
    sinfonia.add_argument("--embedding", required=True, help="H5 embedding file with dataset `data`")
    sinfonia.add_argument("--labels", nargs="*", default=[], help="One or more cell-type CSV files")
    sinfonia.add_argument("--out", required=True, help="Output H5 clustering path")
    sinfonia.add_argument("--batch", action="store_true", help="Use the batch clustering script")
    sinfonia.add_argument("--num", type=int, help="Number of clusters for the batch clustering script")
    common_options(sinfonia)
    sinfonia.set_defaults(func=sinfonia_clustering)

    mlp = subparsers.add_parser("mlp-classification", help="Run the MLP classification workflow")
    mlp.add_argument("--reference", required=True, help="Reference H5 file")
    mlp.add_argument("--query", nargs="+", required=True, help="One or more query H5 files")
    mlp.add_argument("--reference-labels", required=True, help="Reference cell-type CSV")
    mlp.add_argument("--query-labels", nargs="+", required=True, help="One or more query cell-type CSV files")
    mlp.add_argument("--out", required=True, help="Output directory")
    mlp.add_argument("--epochs", type=int, default=10)
    mlp.add_argument("--batch-size", type=int, default=64)
    mlp.add_argument("--lr", type=float, default=1e-2)
    mlp.add_argument("--seed", type=int, default=1)
    common_options(mlp)
    mlp.set_defaults(func=mlp_classification)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
