# 改进版 X-Gaussian 在 CT 数据集下的多视角合成训练流程说明

## 1. 任务目标

本文档描述改进版 X-Gaussian 在 CT 数据集下进行多视角合成的完整训练流程。任务输入为 CT 体数据预处理得到的多角度 X-ray 投影及其 cone-beam 几何参数，训练目标是在稀疏或有限训练视角监督下优化 3D Gaussian 表示，使模型能够在未参与训练的测试视角上合成高质量投影图像。

整体流程可以概括为：

```text
CT pickle 数据
  -> 读取投影与扫描几何
  -> 构建训练/测试相机
  -> 基于体素网格初始化 Gaussian 点云
  -> 随机视角渲染与监督优化
  -> densify/prune 动态调整点云
  -> 测试视角评估 SSIM、PSNR、速度和点数
```

与原始 X-Gaussian 相比，本文方法的重点改进在于：面向 CT 投影图像的结构边界和高频细节，引入结构注意力加权的鲁棒重建损失，并统一在归一化投影域进行训练和评估，使优化过程更加关注骨性边缘、组织边界和投影纹理细节。

## 2. CT 数据读取与相机构建

训练数据以 pickle 文件组织，文件中包含 cone-beam CT 扫描几何、训练投影、验证投影和原始体数据。训练阶段主要使用以下内容：

| 字段 | 作用 |
| --- | --- |
| `DSD` | X-ray 源到探测器距离，用于计算相机 FoV |
| `DSO` | X-ray 源到旋转中心距离，用于构建相机圆轨迹 |
| `nDetector` / `dDetector` | 探测器像素数量和物理间距 |
| `nVoxel` / `dVoxel` | CT 体素数量和物理间距 |
| `train.projections` | 训练视角投影图像 |
| `train.angles` | 训练视角角度 |
| `val.projections` | 测试视角投影图像 |
| `val.angles` | 测试视角角度 |

每个投影视角会被转换为一个相机。相机位姿由扫描角度和 `DSO` 决定，默认采用圆轨迹 cone-beam 扫描模型：X-ray 源围绕体数据中心旋转，每个角度对应一个 camera-to-world 矩阵，再反求 world-to-camera 矩阵得到训练所需的旋转 `R` 和平移 `T`。

FoV 的计算使用探测器尺寸和 `DSD`：

```text
FovX = focal2fov(DSD, detector_width)
FovY = focal2fov(DSD, detector_height)
```

数据读取后会形成三类相机：

| 相机集合 | 来源 | 用途 |
| --- | --- | --- |
| train cameras | `train.projections` 和 `train.angles` | 参与训练优化 |
| test cameras | `val.projections` 和 `val.angles` | 评估新视角合成质量 |
| additional cameras | 随机角度生成的空投影相机 | 用于扩展潜在渲染视角 |

在启用评估模式时，训练集和测试集保持分离；训练过程只从 train cameras 中随机采样视角，测试指标只在 test cameras 上计算。

## 3. 点云初始化

CT 数据具备明确的三维体素空间，因此本文方法不依赖 SfM 点云，而是直接根据 CT 体素网格生成初始 Gaussian 点云。

初始化步骤如下：

1. 根据 `nVoxel`、`dVoxel` 计算 CT 体数据对应的三维物理坐标网格。
2. 使用固定间隔 `interval` 对体素网格进行均匀下采样。
3. 将采样得到的三维坐标作为 Gaussian 初始位置。
4. 为每个点初始化 SH 特征、尺度、旋转和不透明度相关参数。
5. 将初始化点云保存为输入点云文件，并在训练中继续动态增密和剪枝。

默认 `interval=8`。在样例 CT 数据 `1.3.6.1.4.1.9328.50.4.0737` 中，初始点数约为 `90112`。这种初始化方式使点云天然覆盖 CT 体空间，相比随机点云更适合医学体数据投影任务。

## 4. 训练主流程

训练入口读取配置文件后，会合并命令行参数和 YAML 配置，并创建输出目录、日志文件、相机列表和初始 Gaussian 模型。核心训练循环执行 `20000` 次迭代，主要步骤如下。

### 4.1 随机视角采样

每轮训练从 train cameras 中随机弹出一个视角。当视角缓存为空时，重新复制训练相机列表。这样可以保证训练过程中不同角度被反复采样，同时避免每一轮都固定顺序遍历。

```text
if viewpoint_stack is empty:
    viewpoint_stack = train_cameras.copy()
viewpoint_cam = random_pop(viewpoint_stack)
```

### 4.2 前向渲染

当前视角会被送入 Gaussian rasterizer。渲染器根据相机内外参、Gaussian 的三维位置、尺度、旋转、SH 特征和 radiodensity/opacity 参数生成投影图像，同时返回：

| 返回项 | 含义 |
| --- | --- |
| `render` | 当前视角合成投影 |
| `viewspace_points` | 屏幕空间点，用于累积梯度统计 |
| `visibility_filter` | 当前视角可见点标记 |
| `radii` | Gaussian 在屏幕空间的半径 |

训练监督使用相机中的归一化投影图像 `normalized_image`。归一化可以降低不同 CT 投影强度范围差异对优化的影响，使 PSNR、SSIM 等指标在统一尺度上计算。

### 4.3 结构注意力加权损失

原始 X-Gaussian 的重建项主要使用 L1 损失，并结合 SSIM：

```text
loss_original = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
```

本文方法将像素重建项替换为结构注意力加权 Charbonnier 损失：

```text
loss_improved = (1 - lambda_dssim) * Lcharb_weighted
                + lambda_dssim * (1 - SSIM)
```

其中 `lambda_dssim=0.2`，因此 Charbonnier 重建项占主要权重，SSIM 用于约束整体结构相似性。

#### 结构注意力图

结构注意力图由两个部分组成：

1. Sobel 梯度响应：突出投影中的边缘和轮廓。
2. 高频残差：原图与高斯模糊图的差值，用于强调局部细节。

```text
edge = sqrt(SobelX(gt)^2 + SobelY(gt)^2)
high = abs(gt - GaussianBlur(gt))
attention = normalize(edge + hf_lambda * high)
```

默认 `hf_lambda=0.5`、`blur_kernel=3`、`blur_sigma=1.5`。该设计使损失更关注 CT 投影中的骨性结构边缘和细粒度纹理，而不是把所有像素视为同等重要。

#### 自适应 gate 与 warmup

直接在训练早期施加强结构权重容易导致优化不稳定，因此本文方法加入 warmup：

```text
if iteration <= attn_warmup_start:
    warmup_ratio = 0
else:
    warmup_ratio = min(1, (iteration - attn_warmup_start) / attn_warmup_iters)
```

样例配置中：

| 参数 | 数值 | 作用 |
| --- | --- | --- |
| `attn_warmup_start` | `2000` | 2000 iter 前不启用额外结构权重 |
| `attn_warmup_iters` | `5000` | 逐步增强注意力权重 |
| `attn_alpha` | `0.8` | 结构权重最大强度 |
| `attn_adaptive_gate` | `True` | 使用分位数自适应阈值 |
| `attn_gate_quantile` | `0.85` | 关注结构响应最高的区域 |

自适应 gate 使用注意力图的分位数作为阈值，只强化高结构响应区域：

```text
gate_thresh = quantile(attention, attn_gate_quantile)
attn_gate = clamp((attention - gate_thresh) / (1 - gate_thresh), 0, 1)
pixel_weight = 1 + attn_alpha * warmup_ratio * attn_gate * attention
```

最终加权 Charbonnier 损失为：

```text
Lcharb_weighted = sum(pixel_weight * sqrt((render - gt)^2 + eps^2))
                  / sum(pixel_weight)
```

默认 `eps=0.001`。相比 L1，Charbonnier 损失在误差较大区域更平滑，有助于稳定优化；结合结构注意力后，模型会把更多梯度分配给 CT 投影中的关键结构区域。

### 4.4 反向传播与参数更新

每轮计算总损失后执行反向传播，并由优化器更新 Gaussian 参数。训练中包含多组学习率：

| 参数组 | 作用 |
| --- | --- |
| position | Gaussian 三维位置 |
| feature | SH/radiance 相关特征 |
| opacity / radiodensity | 投影强度和透明度相关参数 |
| scaling | Gaussian 尺度 |
| rotation | Gaussian 旋转 |

位置学习率采用指数衰减，样例配置为：

```text
position_lr_init = 0.00019
position_lr_final = 0.0000019
position_lr_delay_mult = 0.01
position_lr_max_steps = 20000
```

训练每 `1000` 次迭代提升一次 SH degree，使模型先学习低频结构，再逐步增强表达能力。

### 4.5 点云增密与剪枝

训练前期会根据屏幕空间梯度和可见性统计动态调整点云：

```text
if iteration < densify_until_iter:
    accumulate viewspace gradient stats
    if iteration > densify_from_iter and iteration % densification_interval == 0:
        densify_and_prune()
    if iteration % opacity_reset_interval == 0:
        reset_opacity()
```

样例 CT 配置中：

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| `densify_from_iter` | `500` | 500 iter 后开始增密 |
| `densify_until_iter` | `8000` | 8000 iter 前执行增密/剪枝 |
| `densification_interval` | `200` | 每 200 iter 调整一次点云 |
| `densify_grad_threshold` | `0.00015` | 增密梯度阈值 |
| `opacity_reset_interval` | `4000` | 每 4000 iter 重置 opacity |
| `radiodensity_reset_interval` | `2000` | 每 2000 iter 重置 radiodensity 相关统计 |

点云增密使细节区域获得更多 Gaussian 表示能力；剪枝则移除低贡献或异常点，控制模型规模和渲染效率。

### 4.6 可视化与日志

训练过程中每 `100` iter 记录结构 gate 阈值和 Charbonnier 重建项：

```text
Iter:xxxx, gate_thresh=..., Lcharb=...
```

当 `mask_vis_interval > 0` 时，会额外保存注意力相关图像：

| 文件类型 | 含义 |
| --- | --- |
| `attn_struct` | Sobel + 高频残差生成的结构注意力 |
| `attn_gate` | 分位数阈值后的结构 gate |
| `pixel_weight` | 实际参与加权损失的像素权重 |

这些可视化结果可以用于检查模型是否真正关注了 CT 投影中的结构边界和高频细节。

## 5. 评估流程

评估在指定迭代执行，默认包含 `100`、`2000`、`20000`。每次评估时，模型在 test cameras 上逐视角渲染，并计算平均 SSIM 和 PSNR。

评估流程如下：

```text
for each test camera:
    render image
    clamp image to [0, 1]
    compare with normalized ground truth
    accumulate SSIM and PSNR
average metrics over all test cameras
record testing speed and total point number
```

评估指标含义：

| 指标 | 含义 |
| --- | --- |
| SSIM | 结构相似性，越高说明合成投影结构越接近真实投影 |
| PSNR | 峰值信噪比，越高说明像素误差越小 |
| Testing Speed | 测试视角平均渲染速度 |
| total_points | 当前 Gaussian 点云规模 |

## 6. 样例训练结果

以下结果来自 CT 样例 `1.3.6.1.4.1.9328.50.4.0737` 的一次完整训练。该结果用于说明训练过程中的指标变化，不作为严格消融实验结论。

| Iteration | SSIM | PSNR | total_points | Testing Speed |
| --- | ---: | ---: | ---: | ---: |
| 100 | 0.7860 | 15.4759 | 90112 | 80.82 fps |
| 2000 | 0.9821 | 36.8315 | 88892 | 74.47 fps |
| 20000 | 0.9912 | 44.9162 | 83568 | 57.88 fps |

从训练日志可以观察到：

1. 早期 `100` iter 时，模型已经能够生成基本投影结构，但 PSNR 仍较低。
2. 到 `2000` iter 时，SSIM 提升到 `0.9821`，说明主要结构已经对齐。
3. 最终 `20000` iter 时，SSIM 达到 `0.9912`，PSNR 达到 `44.9162`，表明测试视角的投影合成质量进一步提升。
4. 点数从初始约 `90112` 调整到最终 `83568`，说明 densify/prune 过程并非单纯增加点数，而是在增密和剪枝之间寻找更有效的点云表示。

## 7. 与原始 X-Gaussian 的代码差异

本文方法保留了原始 X-Gaussian 的 radiative Gaussian splatting 主框架，包括 cone-beam 投影建模、Gaussian rasterizer、SH 表达、动态点云增密和测试视角评估。在此基础上，主要做了以下面向 CT 多视角合成的改进。

### 7.1 损失函数从 L1 重建改为结构注意力加权 Charbonnier

原始训练目标：

```text
loss = (1 - lambda_dssim) * L1(render, gt)
       + lambda_dssim * (1 - SSIM(render, gt))
```

改进后：

```text
loss = (1 - lambda_dssim) * WeightedCharbonnier(render, gt, pixel_weight)
       + lambda_dssim * (1 - SSIM(render, gt))
```

该改动使训练不再对所有像素平均施加相同重建约束，而是对 CT 投影中的边缘、高频纹理和结构边界区域施加更强监督。

### 7.2 新增结构注意力图

新增 `structural_attention_map`，通过 Sobel 边缘和高频残差构造结构响应：

```text
attention = normalize(edge + hf_lambda * high_frequency)
```

这相当于在监督信号中显式编码 CT 投影的结构先验，有利于多视角合成时保持解剖结构边界的清晰度。

### 7.3 新增加权 Charbonnier 重建项

新增 `weighted_charbonnier_loss`，将像素误差写成平滑鲁棒形式：

```text
sqrt(error^2 + eps^2)
```

相比 L1，该形式在误差接近零时更平滑，在高误差区域也更稳定；与结构权重结合后，可以把优化重点放在更有诊断意义的结构区域。

### 7.4 新增 adaptive gate 和 warmup

注意力权重不是从训练一开始就直接满强度启用，而是经过 warmup 逐步增加。自适应 gate 则使用注意力图分位数筛选高响应区域，避免低结构背景区域被过度加权。

这一设计降低了训练早期的不稳定性，并使后期优化更集中于 CT 投影中的关键结构。

### 7.5 新增注意力可视化

训练中可按固定间隔输出 `attn_struct`、`attn_gate`、`pixel_weight`。这些图像可以用于分析注意力是否聚焦在骨性结构边缘、组织边界和高频细节区域，也便于调试 `attn_alpha`、`attn_gate_quantile` 等超参数。

### 7.6 新增可复现实验设置

训练入口新增 `--seed` 参数，并在随机数初始化中同时设置 Python、NumPy、PyTorch 和 CUDA 随机种子；同时关闭 cuDNN benchmark 并启用 deterministic 行为。这样可以减少多次训练之间由随机初始化和随机视角采样带来的差异。

### 7.7 CT 训练超参数调整

针对 CT 投影任务，样例配置将训练长度设为 `20000` iter，并采用更适合 CT 体空间初始化的点云调整策略：

| 参数 | 改进版设置 |
| --- | ---: |
| `iterations` | 20000 |
| `densification_interval` | 200 |
| `opacity_reset_interval` | 4000 |
| `radiodensity_reset_interval` | 2000 |
| `densify_from_iter` | 500 |
| `densify_until_iter` | 8000 |
| `random_background` | False |

这些设置使训练前期有足够机会进行点云结构调整，后期则更专注于稳定优化 Gaussian 参数和投影细节。

## 8. 小结

改进版 X-Gaussian 的 CT 多视角合成训练流程以 CT pickle 数据为输入，通过 cone-beam 几何构建相机，以体素网格初始化 Gaussian 点云，并在训练中结合结构注意力加权 Charbonnier 损失和 SSIM 约束进行优化。

相对原始 X-Gaussian，本文方法的核心改进集中在 CT 投影监督信号的设计：使用 Sobel 边缘和高频残差生成结构注意力，再通过 adaptive gate、warmup 和加权 Charbonnier 损失将更多优化权重分配给关键结构区域。样例结果显示，在 `20000` 次迭代后，测试视角 SSIM 达到 `0.9912`，PSNR 达到 `44.9162`，说明该训练流程能够在 CT 数据集下获得较高质量的多视角合成结果。
