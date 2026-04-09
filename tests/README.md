# Tests

This folder contains a self-contained regression test for the streaming PCA refactor in `InTAct/intact.py`.

## What it checks

- The streaming implementation computes the same mean and PCA components as a reference implementation that materializes activations on a small toy model.
- The test only uses CPU execution and does not require pytest.

## Requirements

- Python 3.10+ recommended
- PyTorch installed in the target environment

## Run

From the repository root:

```bash
python tests/test_streaming_pca.py
```

Expected output:

```text
streaming_pca_ok
```

If you want to run it from another machine or environment, copy the repository and ensure the repo root is on `PYTHONPATH` or run the command from the root directory as shown above.