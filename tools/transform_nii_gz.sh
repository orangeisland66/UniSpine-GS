#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input.nii.gz> <output.pickle> [config.yaml] [extra converter args...]"
    echo "Example: $0 /abs/path/case.nii.gz /abs/path/UniSpine-GS/data/case.pickle"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_PATH="$1"
OUTPUT_PATH="$2"
shift 2

CONFIG_PATH="${SCRIPT_DIR}/configs/ctspine1k_spine.yaml"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    CONFIG_PATH="$1"
    shift 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_nii_gz_to_pickle.py" \
    --input_nii_gz "${INPUT_PATH}" \
    --output_pickle "${OUTPUT_PATH}" \
    --config "${CONFIG_PATH}" \
    "$@"
