# UniSpine-GS

This repository contains the code for **UniSpine-GS: An Efficient Physics-Aware
Gaussian Framework for Cross-Modality Multi-view Spine Image Synthesis**.  The
codebase is developed from [X-Gaussian](https://github.com/caiyuanhao1998/X-Gaussian)
and keeps the projection rendering pipeline used by
[SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF).

## 1. Environment

Create the UniSpine-GS training environment from a fresh clone:

```bash
git clone https://github.com/orangeisland66/UniSpine-GS.git
cd UniSpine-GS
git submodule update --init --recursive submodules/diff-gaussian-rasterization/third_party/glm
conda env create --file environment.yml
conda activate unispine_gs
```

After activating the environment, verify that the environment-provided compiler is
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

Then install the two in-repository extension modules:

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

These extensions are installed after `conda env create` because their setup
scripts import PyTorch during compilation.


## 2. Prepare Volume Data

### CT NIfTI Data

Download CTSpine3D cases from [Baidu Disk](https://pan.baidu.com/s/1jyQMXGehP1Eaoec_Kieqag?pwd=fa9h)

The original case files are NIfTI volumes such as:

```text
<ctspine3d_root>/<case_id>.nii.gz
```
Convert each case to the pickle format expected by UniSpine-GS with the
in-repository converter:

```bash
bash tools/transform_nii_gz.sh \
  /abs/path/to/<case_id>.nii.gz \
  data/<case_id>.pickle
```


### Ultrasound VOL Data

Under the guidance of professional clinicians and following established clinical practice guidelines, we curated a formal dataset comprising 242 three-dimensional fetal spine ultrasound volumes from 102 patients, with the majority of the data acquired using GE Voluson E8/E10 ultrasound systems. All data were de-identified and anonymized prior to use. For each volume, multi-view projection images were generated using a GPU-accelerated differentiable DRR operator under known cone-beam geometry. Specifically, 100 projection views were generated for each volume, of which 50 views were used for training and the remaining 50 views were reserved for validation.

The open-source release of the dataset is actively underway and is currently in the stage of data curation and publication preparation. You can view the preview images in [FeSpine_3D_preview](https://github.com/orangeisland66/UniSpine-GS/tree/master/FeSpine_3D_preview).

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

## 3. Training Config

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

## 4. Train One Case

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

## 5. Batch Train and Evaluate Cases


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


## 6. Render Trained Views

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

The experimental results reported in the paper, including SSIM, PSNR, training time, and inference time, are averaged over the results from the four .nii.gz datasets in the CTSpine3D_example folder. SSIM, PSNR, and training time are obtained during the training stage, while inference time is measured during the testing stage using render.py.

## 7. Citation

If you use this codebase in your research, please cite the following paper:


## Acknowledgements

This codebase builds on X-Gaussian, SAX-NeRF, 3D Gaussian Splatting, and the
CTSpine3D dataset. Please follow the licenses and usage requirements of the
original projects and dataset when using this repository.
