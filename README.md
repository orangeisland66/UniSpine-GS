# UniSpine-GS

This repository contains the code for **UniSpine-GS: An Efficient Physics-Aware
Gaussian Framework for Cross-Modality Multi-view Spine Image Synthesis**.  The
codebase is developed from [X-Gaussian](https://github.com/caiyuanhao1998/X-Gaussian)
and keeps the projection rendering pipeline used by
[SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF).

Volume data are first converted to the UniSpine-GS pickle format and then
trained with the same UniSpine-GS commands.  The repository provides converters
for CT NIfTI files (`.nii.gz`) and ultrasound KRETZ volume files (`.vol`).

## 1. Pipeline Overview

```text
CT .nii.gz or ultrasound .vol
  -> tools/transform_nii_gz.sh or tools/transform_vol.sh
  -> data/<case_id>.pickle
  -> tools/generate_data_yaml_configs.py
  -> config/<case_id>.yaml
  -> train.py or batch_train_eval.py
  -> output/<case_id>/point_cloud/iteration_*/point_cloud.ply
  -> render.py
  -> output/<case_id>/{train,test}/ours_*/renders/*.png
```

The converted pickle stores the input volume, cone-beam geometry, training
projections, validation projections, and their scanning angles.  During training,
`scene/dataset_readers.py` reads the pickle, builds circular cone-beam camera
poses from `DSO` and each projection angle, and initializes the Gaussian point
cloud by uniformly sampling the voxel grid with `interval` spacing.

With `--eval`, `train.projections` are used for optimization and
`val.projections` are kept as held-out test views.  Without `--eval`, validation
views are merged into the training cameras and no test metric is reported.

## 2. Environment

Create the UniSpine-GS training environment from a fresh clone:

```bash
git clone https://github.com/orangeisland66/UniSpine-GS.git
cd UniSpine-GS
git submodule update --init --recursive submodules/diff-gaussian-rasterization/third_party/glm
conda env create --file environment.yml
conda activate unispine_gs
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
license.  Build it from the repository copy before running the conversion
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

These extensions are installed after `conda env create` because their setup
scripts import PyTorch during compilation.


## 3. Prepare Volume Data

### CT NIfTI Data

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
You can also directly download .pickle files of CTSpine3D from [Baidu Disk](https://pan.baidu.com/s/1fCvqB8BjbxYjyLFoqWlx9w?pwd=g3ut)(code: g3ut) and put them in ./data folder.

### Ultrasound VOL Data

Under the guidance of professional clinicians and following established clinical practice guidelines, we curated a formal dataset comprising 242 three-dimensional fetal spine ultrasound volumes from 102 patients, with the majority of the data acquired using GE Voluson E8/E10 ultrasound systems. All data were de-identified and anonymized prior to use. For each volume, multi-view projection images were generated using a GPU-accelerated differentiable DRR operator under known cone-beam geometry. Specifically, 100 projection views were generated for each volume, of which 50 views were used for training and the remaining 50 views were reserved for validation.

The open-source release of the dataset is actively underway and is currently in the stage of data curation and publication preparation.

However, you can currently train the model using your own ultrasound dataset in the .vol format. Ultrasound KRETZ `.vol` files can be converted with:

```bash
bash tools/transform_vol.sh \
  /abs/path/to/<case_id>.vol \
  data/<case_id>.pickle
```

After conversion, keep pickle files directly under `UniSpine-GS/data/`:

```sh
  |--data
      |--case_001.pickle
      |--case_002.pickle
      |--case_003.pickle
      ...
```

The config generator and `batch_train_eval.py` scan only `data/*.pickle` at the
first directory level.  Files under subdirectories are ignored.

## 4. Training Config

Each pickle should have a matching YAML config under `config/`.  Generate
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

## 5. Train One Case

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

The log reports the test-view SSIM, PSNR, rendering speed, and final Gaussian
point count.

## 6. Batch Train and Evaluate Cases


Train every pickle in `data/` sequentially:

```bash
python batch_train_eval.py \
  --project_root . \
  --data_dir data \
  --config_dir config \
  --output_dir output \
  --gpu_id 0
```

You can also use the helper script:

```bash
bash train.sh
```


## 7. Render Trained Views

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

## 8. Citation

If you use this codebase in your research, please cite the following paper:



## Acknowledgements

This codebase builds on X-Gaussian, SAX-NeRF, 3D Gaussian Splatting, and the
CTSpine1K dataset.  Please follow the licenses and usage requirements of the
original projects and dataset when using this repository.
