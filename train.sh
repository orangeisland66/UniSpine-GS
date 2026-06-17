#!/bin/bash
set -e

# Switch to the script directory so relative paths stay stable.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Assumes the required conda environment has already been activated.

echo "========================================"
echo "Running batch training and evaluation for all top-level data/*.pickle files"
echo "========================================"

if command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
else
	echo "Error: python/python3 not found in PATH"
	exit 1
fi

"${PYTHON_BIN}" batch_train_eval.py --project_root "." --data_dir "data"

echo "========================================"
echo "Batch training and evaluation completed."
echo "Summary saved to: output/batch_eval_summary.json"
echo "Summary saved to: output/batch_eval_summary.txt"
echo "========================================"
