# Contributing

Thanks for improving scMultiBench. The repository is organized as a benchmark script collection, so contributions should preserve script reproducibility and make assumptions explicit.

## Before Opening a Pull Request

1. Add or update documentation for new inputs, outputs, and dependencies.
2. Keep generated files out of Git, including `__pycache__`, `.pyc`, `build/`, `dist/`, and `*.egg-info/`.
3. Add or update `metadata/methods.yaml` when changing method scripts.
4. Prefer small demo data for examples. Large benchmark data should be linked externally with a checksum or version note.
5. Run the lightweight checks:

```bash
python -m unittest discover -s tests
python scripts/validate_inputs.py classification \
  --reference data/classification/demo_data/data1.h5 \
  --query data/classification/demo_data/data2.h5 \
  --reference-labels data/classification/demo_data/cty1.csv \
  --query-labels data/classification/demo_data/cty2.csv
```

## Adding a Method

For a new method, include:

- The method script under `tools_scripts/<MethodName>/`.
- The exact method version or commit used.
- The original installation instructions or a method-specific environment file.
- The expected input format and output format.
- A registry entry in `metadata/methods.yaml`.

## Dependency Guidance

Avoid forcing all methods into one environment. Many methods depend on incompatible versions of Python, R, PyTorch, TensorFlow, Seurat, scanpy, or scvi-tools. Use the common evaluation environments in `envs/` for validation and metrics, and use method-specific environments for full benchmark runs.
