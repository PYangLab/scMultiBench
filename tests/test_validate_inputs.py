import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ValidateInputsTests(unittest.TestCase):
    def validate(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_inputs.py"), *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_labels_accepts_r_style_rowname_column(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            labels = Path(tempdir) / "cty.csv"
            labels.write_text('"","x"\n"1","B cell"\n"2","T cell"\n', encoding="utf-8")
            result = self.validate("labels", "--labels", str(labels), "--expected-rows", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK labels: 2 rows", result.stdout)

    def test_labels_reports_expected_row_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            labels = Path(tempdir) / "cty.csv"
            labels.write_text("x\nB cell\n", encoding="utf-8")
            result = self.validate("labels", "--labels", str(labels), "--expected-rows", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
