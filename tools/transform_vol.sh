#!/usr/bin/env bash
set -euo pipefail

# Thread-parallel work inside each conversion process.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-20}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-20}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-20}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input.vol|input_dir> <output.pickle|output_dir> [config.yaml] [extra converter args...]"
    echo "Example(file): $0 /path/to/file.vol data/file.pickle"
    echo "Example(dir):  $0 /path/to/in_dir data"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_PATH="$1"
OUTPUT_PATH="$2"
shift 2

CONFIG_PATH="${SCRIPT_DIR}/configs/fespine3d_spine.yaml"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    CONFIG_PATH="$1"
    shift 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_vol_to_pickle.py"

if [ ! -f "${CONVERT_SCRIPT}" ]; then
    echo "Converter not found: ${CONVERT_SCRIPT}"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Config not found: ${CONFIG_PATH}"
    exit 1
fi

vis_dir_for_pickle() {
    local out_pkl="$1"
    local base="${out_pkl%.*}"
    echo "${base}_vis"
}

convert_one() {
    local in_vol="$1"
    local out_pkl="$2"
    shift 2
    local vis_dir
    vis_dir="$(vis_dir_for_pickle "${out_pkl}")"

    mkdir -p "$(dirname "${out_pkl}")" "${vis_dir}"
    echo "[Convert] ${in_vol} -> ${out_pkl}"
    "${PYTHON_BIN}" "${CONVERT_SCRIPT}" \
        --input_vol "${in_vol}" \
        --output_pickle "${out_pkl}" \
        --config "${CONFIG_PATH}" \
        --vis_dir "${vis_dir}" \
        "$@"
}

if [ -f "${INPUT_PATH}" ]; then
    convert_one "${INPUT_PATH}" "${OUTPUT_PATH}" "$@"
    echo "Transform completed: ${INPUT_PATH} -> ${OUTPUT_PATH}"
elif [ -d "${INPUT_PATH}" ]; then
    mkdir -p "${OUTPUT_PATH}"
    found=0
    while IFS= read -r -d '' f; do
        found=1
        rel="${f#${INPUT_PATH}/}"
        rel_noext="${rel%.vol}"
        out_file="${OUTPUT_PATH}/${rel_noext}.pickle"
        convert_one "${f}" "${out_file}" "$@"
    done < <(find "${INPUT_PATH}" -type f -name "*.vol" -print0)

    if [ "${found}" -eq 0 ]; then
        echo "No .vol files found under: ${INPUT_PATH}"
        exit 1
    fi
    echo "Batch transform completed: ${INPUT_PATH} -> ${OUTPUT_PATH}"
else
    echo "Input path not found: ${INPUT_PATH}"
    exit 1
fi
