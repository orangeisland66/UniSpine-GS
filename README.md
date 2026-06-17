# UniSpine-GS

This repository contains the code for **UniSpine-GS: An Efficient Physics-Aware
Gaussian Framework for Cross-Modality Multi-view Spine Image Synthesis**.  The
codebase is developed from [X-Gaussian](https://github.com/caiyuanhao1998/X-Gaussian)
and currently keeps the X-ray/CT projection rendering pipeline used by
[SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF).

This README currently documents only the **CT dataset training pipeline**.  The
CT data are prepared from [CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K)
NIfTI files (`.nii.gz`), converted to X-Gaussian-compatible pickle files, and
then trained with UniSpine-GS.

## 1. Pipeline Overview

```text
CTSpine1K .nii.gz
  -> tools/transform_nii_gz.sh
  -> UniSpine-GS data/<case_id>.pickle
  -> tools/generate_data_yaml_configs.py
  -> config/<case_id>.yaml
  -> train.py or batch_train_eval.py
  -> output/<case_id>/point_cloud/iteration_*/point_cloud.ply
  -> render.py
  -> output/<case_id>/{train,test}/ours_*/renders/*.png
```

The converted pickle stores the CT volume, cone-beam geometry, training
projections, validation projections, and their scanning angles.  During training,
`scene/dataset_readers.py` reads the pickle, builds circular cone-beam camera
poses from `DSO` and each projection angle, and initializes the Gaussian point
cloud by uniformly sampling the CT voxel grid with `interval` spacing.

With `--eval`, `train.projections` are used for optimization and
`val.projections` are kept as held-out test views.  Without `--eval`, validation
views are merged into the training cameras and no test metric is reported.

## 2. Environment

Create the UniSpine-GS training environment from the project root:

```bash
git clone --recursive https://github.com/orangeisland66/UniSpine-GS.git
cd UniSpine-GS
conda env create --file environment.yml
conda activate unispine_gs
```

If you cloned the repository without `--recursive`, initialize all nested
submodules before installing the CUDA extensions:

```bash
git submodule update --init --recursive
```

Verify that PyTorch imports correctly before building the local CUDA extensions:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The local CUDA extension modules are compiled with `nvcc` and CUDA development
headers.  The CUDA compiler version must match the CUDA version used by PyTorch.
`environment.yml` installs `cuda-nvcc=11.6`, `cuda-cudart=11.6`, and
`cuda-cudart-dev=11.6` for this purpose.  It also installs
`libcurand-dev=10.2.9.55`, which TIGRE needs for `curand_kernel.h`.  After
activating the environment, verify that the environment-provided compiler is
selected:

```bash
which nvcc
nvcc --version
```

The reported CUDA release should be `11.6`.  If it reports a different system
CUDA version, reactivate the conda environment or place `$CONDA_PREFIX/bin`
before the system CUDA path in `PATH`.

TIGRE source is vendored in `third_party/TIGRE-2.3` under the BSD 3-Clause
license.  Build it from the repository copy before running the CT conversion
scripts:

```bash
cd third_party/TIGRE-2.3/Python
CUDA_HOME="$CONDA_PREFIX" python setup.py develop
cd ../../..
python -c "import tigre; print('TIGRE OK')"
```

Setting `CUDA_HOME="$CONDA_PREFIX"` makes TIGRE use the CUDA compiler and
runtime libraries installed in the conda environment instead of a system CUDA
installation.

Then install the two in-repository extension modules:

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

These extensions are intentionally installed after `conda env create` rather
than from the `pip:` section of `environment.yml`. Their setup scripts import
PyTorch while building, so installing them after the environment is activated
makes failures easier to diagnose and avoids creating a partially configured
environment.

If environment creation previously failed while installing these submodules,
remove the incomplete environment and recreate it:

```bash
conda env remove -n unispine_gs
conda env create --file environment.yml
conda activate unispine_gs
```

The local training helper `train.sh` assumes this environment has already been
activated and uses the `python` executable from the current shell.

The NIfTI-to-pickle conversion script is included in this repository under
`tools/`.  It uses the SAX-NeRF data generation convention and the vendored
TIGRE build above.  The Python dependencies used by the converter (`nibabel`,
`scipy`, `PyYAML`, and `imageio`) are included in `environment.yml`.

## 3. Prepare CTSpine1K Data

Download CTSpine1K cases from:

```text
https://github.com/MIRACLE-Center/CTSpine1K
```

The original case files are NIfTI volumes such as:

```text
<ctspine1k_root>/<case_id>.nii.gz
```

Convert each case to the pickle format expected by UniSpine-GS with the
in-repository converter:

```bash
bash tools/transform_nii_gz.sh \
  /abs/path/to/<case_id>.nii.gz \
  data/<case_id>.pickle
```

Example:

```bash
bash tools/transform_nii_gz.sh \
  /abs/path/to/HN_P001.nii.gz \
  data/HN_P001.pickle \
  --vis_dir data/vis/HN_P001
```

The current `transform_nii_gz.sh` calls
`tools/convert_nii_gz_to_xgaussian_pickle.py`, which:

- loads the `.nii.gz` volume with nibabel;
- converts HU values to attenuation when `convert: True`;
- resizes the volume according to `tools/configs/ctspine1k_spine.yaml`;
- simulates cone-beam DRR projections with TIGRE;
- writes train and validation projections into one pickle file using pickle
  protocol 4.

By default, `tools/configs/ctspine1k_spine.yaml` generates `50` training
views and `50` validation views over a `180` degree trajectory.  Validation
views are placed at the midpoints between adjacent training views.

After conversion, keep CT pickle files directly under `UniSpine-GS/data/`:

```text
UniSpine-GS/
  data/
    case_001.pickle
    case_002.pickle
    case_003.pickle
```

The config generator and `batch_train_eval.py` scan only `data/*.pickle` at the
first directory level.  Files under subdirectories such as `data/ultrasound/`
are ignored.

## 4. CT Training Config

Each CT pickle should have a matching YAML config under `config/`.  Generate
configs from all pickle files directly under `data/` with:

```bash
python tools/generate_data_yaml_configs.py
```

The script writes `config/<case_id>.yaml` for each `data/<case_id>.pickle`.
The `scene` name is taken from the pickle filename, and `source_path` points to
the converted pickle. Existing YAML files are skipped by default; use
`--overwrite` to regenerate them from the template:

```bash
python tools/generate_data_yaml_configs.py --overwrite
```

## 5. Train One CT Case

Run one case from the UniSpine-GS root:

```bash
python train.py \
  --config config/<case_id>.yaml \
  --eval \
  --model_path output/<case_id> \
  --gpu_id 0
```

Example:

```bash
python train.py \
  --config config/case_001.yaml \
  --eval \
  --model_path output/case_001 \
  --gpu_id 0
```

Training evaluates held-out validation views at the default iterations
`100`, `2000`, and `20000`.  The final point cloud is saved at:

```text
output/<case_id>/point_cloud/iteration_20000/point_cloud.ply
```

Training logs and metrics are written to:

```text
output/<case_id>/log.txt
output/<case_id>/cfg_args
output/<case_id>/cameras.json
output/<case_id>/input.ply
```

The log reports the CT test-view SSIM, PSNR, rendering speed, and final Gaussian
point count.

## 6. Batch Train and Evaluate CT Cases


Train every pickle in `data/` sequentially:

```bash
python batch_train_eval.py \
  --project_root . \
  --data_dir data \
  --config_dir config \
  --output_dir output \
  --gpu_id 0
```

If the conda environment is already active, you can use the helper script:

```bash
bash train.sh
```

`train.sh` changes into the repository root, uses the current environment's
Python interpreter, and runs `batch_train_eval.py` over top-level
`data/*.pickle` files.

Batch behavior:

- scans `data/*.pickle`;
- uses `config/<case_id>.yaml` generated by `tools/generate_data_yaml_configs.py`;
- falls back to a minimal config with only `scene`, `source_path`, and
  `iterations: 20000` if no config exists;
- trains with `python train.py --config ... --eval --model_path output/<case_id>`;
- removes an existing `output/<case_id>` directory before retraining that case;
- parses each `log.txt` and writes summary metrics.

Batch summaries are saved to:

```text
output/batch_eval_summary.json
output/batch_eval_summary.txt
output/batch_eval_summary.csv
```

Avoid running multiple CT trainings in parallel with the same `data/` directory:
the scene loader regenerates `data/points3d.ply` during initialization.
Sequential single-case or batch training is safe.

## 7. Render Trained CT Views

After training, render the trained Gaussian model with:

```bash
python render.py \
  --model_path output/<case_id> \
  --iteration -1 \
  --skip_train
```

`--iteration -1` loads the latest saved point cloud.  `--skip_train` renders
only the held-out test views.  Outputs are written to:

```text
output/<case_id>/test/ours_<iteration>/renders/
output/<case_id>/test/ours_<iteration>/gt/
```

To render both training and test views, omit `--skip_train`.


## Acknowledgements

This codebase builds on X-Gaussian, SAX-NeRF, 3D Gaussian Splatting, and the
CTSpine1K dataset.  Please follow the licenses and usage requirements of the
original projects and dataset when using this repository.
