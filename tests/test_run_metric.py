import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunMetricTests(unittest.TestCase):
    def run_metric(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_metric.py"), *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_scib_embedding_dry_run_dispatches_existing_script(self) -> None:
        result = self.run_metric(
            "scib-embedding",
            "--embedding",
            "embedding.h5",
            "--labels",
            "cty1.csv",
            "cty2.csv",
            "--cluster",
            "cluster.h5",
            "--batch-cluster",
            "batch_cluster.h5",
            "--out",
            "results/demo",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evaluation_pipelines/scib_metrics/scib_metrics.py", result.stdout)
        self.assertIn("--data_path embedding.h5", result.stdout)
        self.assertIn("--cty_path cty1.csv cty2.csv", result.stdout)

    def test_mlp_classification_dry_run_sets_script_cwd(self) -> None:
        result = self.run_metric(
            "mlp-classification",
            "--reference",
            "data1.h5",
            "--query",
            "data2.h5",
            "--reference-labels",
            "cty1.csv",
            "--query-labels",
            "cty2.csv",
            "--out",
            "results/classification/demo",
            "--epochs",
            "2",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evaluation_pipelines/classification/MLP_classification", result.stdout)
        self.assertIn("--epochs 2", result.stdout)

    def test_sinfonia_batch_dry_run_uses_num(self) -> None:
        result = self.run_metric(
            "sinfonia-clustering",
            "--embedding",
            "embedding.h5",
            "--out",
            "batch_cluster.h5",
            "--batch",
            "--num",
            "3",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SINFONIA_batch.py", result.stdout)
        self.assertIn("--num 3", result.stdout)


if __name__ == "__main__":
    unittest.main()
