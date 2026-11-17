# MindSpore Rep-Mamba

面向超高清视频/图像超分辨率的 **Rep-Mamba 轻量化模型的昇思 MindSpore 实现**。

本仓库是开放合作研究课题 **「基于 MindSpore 框架的超高清 8K 时-空视频超分辨率研究」（课题编号 GML-KF-24-09）** 的核心算法开源实现，用于验证重参数化卷积与状态空间模型（State-Space Model, SSM）融合的轻量化超分方案在国产深度学习框架下的可复现性与可部署性。

- 课题承担单位：哈尔滨工业大学
- 资助：Open Research Fund from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), Grant **GML-KF-24-09**

---

## 1. 项目简介

Rep-Mamba 在图像/视频超分辨率任务中把**可微重参数化卷积（RepConv）**与 **Mamba 状态空间结构**结合：训练阶段用多分支结构增强特征表达，推理阶段等效融合为单一卷积，从而在不增加推理计算量的条件下提升局部细节保持能力；跨尺度状态传播负责多层次时空特征融合，轻量渐进融合模块负责特征的高效聚合，整体在参数量与计算量极低的条件下取得较优的精度—效率折中。

本仓库为该方法的**纯 MindSpore 实现**：选择性扫描采用普通 MindSpore 张量算子与自动微分实现，模型运行期**不导入、不调用** `torch`、`mamba_ssm` 或 `selective_scan_cuda`。

## 2. 主要特性

- **纯 MindSpore 算子实现**，无任何 CUDA 扩展依赖；支持 `PYNATIVE_MODE` 与 `GRAPH_MODE`
- **RepConv 重参数化卷积**：训练阶段为 3×3 深度可分离卷积 + BN、1×1 卷积、identity 三分支与外层 BN，`switch_to_deploy()` 后按冻结的 BN 统计量融合为单一 3×3 深度可分离卷积
- **SS2D 四方向状态空间扫描**，并配局部高频分支
- **CCmamba 跨尺度状态传播**：在多个尺度上传递与对齐状态特征，实现层级空间依赖建模
- **RHTB 层级 Transformer 块**，支持任意分辨率的 patch 展开/回贴
- **选择性扫描双后端**：`reference`（逐时刻递推，数值标准答案）与 `cpu_optimized`（Hillis-Steele 前缀扫描 + 精确逆向 bprop），FP32 梯度误差优于 `1e-4`
- **多尺度重建头**：×2 / ×3 / ×4
- **完整单元测试**：8 组测试覆盖前向、反向梯度、跨尺度扫描往返、跨扫描/合并与 PyTorch 数值一致性

## 3. 目录结构

```
mindspore_repmamba/
├── model/
│   ├── repconv.py          # RepConv 重参数化卷积（训练多分支 / 推理单分支）
│   └── repmamba_ms.py      # SS2D、MambaLayer、CCmamba、HierarchicalTransformerBlock、RHTB、SMSR
├── ops/
│   ├── basic.py            # PatchEmbed / PatchUnEmbed / ChannelAttention / CAB / MLP / ConvFFN / Upsample
│   ├── scans.py            # CrossScan、CrossMerge、MultiScan、跨尺度扫描与合并
│   └── selective_scan.py   # 选择性扫描算子（reference / cpu_optimized 双后端）
├── archs/                  # 网络装配
├── datasets/               # 配对数据加载与数据变换
├── tools/
│   ├── benchmark_selective_scan.py  # 选择性扫描前向/反向性能基准与显存采样
│   ├── compare_pytorch.py           # 与 PyTorch 参考实现的数值一致性对比
│   ├── convert_weights.py           # PyTorch → MindSpore 权重转换
│   ├── pytorch_reference.py         # 扫描/合并算子的 PyTorch 参考实现
│   ├── performance_report.py        # 基准结果汇总与报告生成
│   └── cpu.py                       # CPU 线程与后端配置
├── tests/                  # 8 组单元测试
├── train.py                # 最小 FP32 训练入口（L1 + Adam）
├── eval.py                 # 前向冒烟测试入口
├── losses.py               # L1 / MSE / Charbonnier / Edge 损失与 PSNR
├── SELECTIVE_SCAN_PERFORMANCE.md    # 选择性扫描 CPU 性能实测报告
└── LICENSE                 # Apache-2.0
```

## 4. 环境依赖

| 项目 | 版本 |
|---|---|
| Python | 3.8 |
| MindSpore | 2.0.0 及以上 |
| NumPy | 任意兼容版本 |
| 硬件 | CPU 可完整运行（12 GiB 内存环境已验证）；昇腾 NPU 需配套 CANN 7.0 及以上 |

```bash
pip install mindspore==2.0.0 numpy
```

> 说明：切换 `PYNATIVE_MODE` 与 `GRAPH_MODE` 时 MindSpore 会保留已编译图，建议分别独立进程运行，避免图缓存叠加超出内存。

## 5. 快速开始

在仓库根目录的**上一级**执行（本目录本身即 Python 包）：

```bash
# 1) 单元测试
python -m pytest -q mindspore_repmamba/tests

# 2) 前向冒烟测试（默认 ×4，64×64 输入）
python -m mindspore_repmamba.eval --scale 4 --height 64 --width 64

# 3) 选择性扫描性能基准
python -m mindspore_repmamba.tools.benchmark_selective_scan --quick

# 4) 与 PyTorch 参考实现对比
python -m mindspore_repmamba.tools.compare_pytorch

# 5) 最小训练步示例
python -m mindspore_repmamba.train --scale 4 --lr 1e-4
```

选择性扫描后端通过 `--selective-scan-backend` 选择，取值为 `reference`（默认，安全）或 `cpu_optimized`。

```python
import mindspore as ms
from mindspore_repmamba.model import SMSR

net = SMSR(upscale=4, selective_scan_backend="reference")
y = net(ms.ops.zeros((1, 3, 64, 64), ms.float32))
```

## 6. 与论文模块的对应关系

| 论文中的模块 | 代码位置 |
|---|---|
| RepConv 重参数化卷积 | `model/repconv.py` |
| 跨尺度状态传播（CSSP） | `model/repmamba_ms.py` 中的 `CCmamba` |
| 状态空间主干与四方向扫描 | `model/repmamba_ms.py` 的 `SS2D`、`MambaLayer`；`ops/scans.py` |
| 轻量渐进特征融合（LPFM） | `ops/basic.py` 的 `ConvFFN`、`CAB`、`MLP` |
| 选择性扫描算子 | `ops/selective_scan.py` |
| 多尺度重建头 | `ops/basic.py` 的 `Upsample`、`UpsampleOneStep` |

## 7. 选择性扫描性能实测（CPU）

环境：MindSpore 2.0.0 CPU 后端、Python 3.8.20、12 GiB 内存、PYNATIVE_MODE、FP32，计时前以 `asnumpy()` 强制同步。

| B | H | D | K | reference 前向 | optimized 前向 | 前向加速 | reference 反向 | optimized 反向 | 反向加速 | 前向误差 | 梯度误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 48 | 4 | 86.7 ms | 12.6 ms | 6.90× | 162 ms | 277 ms | 0.59× | 3.81e-6 | 2.29e-5 |
| 1 | 16 | 48 | 4 | 382 ms | 18.1 ms | 21.1× | 5290 ms | 989 ms | 5.35× | 9.54e-6 | 3.81e-5 |
| 2 | 16 | 96 | 8 | 367 ms | 40.0 ms | 9.18× | 6549 ms | 1217 ms | 5.38× | 7.63e-6 | 5.34e-5 |
| 1 | 32 | 96 | 8 | 1998 ms | 82.3 ms | 24.3× | 3791 ms | 4484 ms | 0.85× | 7.63e-6 | 9.16e-5 |
| 1 | 64 | 48 | 4 | 8204 ms | 182 ms | 45.2× | — | — | — | 1.34e-5 | — |

完整基准数据、复杂度分析与内存上限说明见 [`SELECTIVE_SCAN_PERFORMANCE.md`](SELECTIVE_SCAN_PERFORMANCE.md)。

## 8. 模型规模与精度

Rep-Mamba 参数量 0.785 M、计算量 10.92 GFLOPs。在 ×4 放大任务上，相较 FMSR、MambaIR-L 分别减少约 93.1% 参数量与 92.9% 计算量，推理速度提升 3.69 倍，在 PSNR/SSIM 上保持较优的精度—效率折中。评测数据集包括 AID、DOTA、DIOR、NWPU-RESISC45 等。

## 9. 昇腾平台适配说明

本仓库提供与硬件无关的纯 MindSpore 实现，可在 CPU 上完整运行与复现。面向昇腾 AI 处理器与 CANN 异构计算架构的算子适配、精度与性能验证数据，随课题结题验收材料提交，未包含在本源码仓库中。

## 10. 单元测试覆盖

| 测试文件 | 覆盖内容 |
|---|---|
| `test_basic.py` | Patch 展开/回贴往返一致性、PixelShuffle 通道顺序、基础张量形状 |
| `test_blocks.py` | SSM 块形状与数值有限性、层级块 GRAPH_MODE 前向 |
| `test_cross_scan.py` | CrossScan 与 PyTorch 一致、MultiScan 可逆性、跨尺度扫描往返 |
| `test_cross_merge.py` | CrossMerge 与 PyTorch 一致、四路扫描合并 |
| `test_selective_scan.py` | 多组参数下的数值一致性、GRAPH_MODE 与梯度、双后端一致性、不同 B/C 布局的反向、精确 bprop |
| `test_gradients.py` | 网络整体单次反向、选择性扫描梯度与 PyTorch 对齐 |
| `test_network_forward.py` | ×4 前向（64×64）、×2/×3 输出形状、reference 后端 |
| `test_repconv.py` | 部署模式与冻结训练图等价性、RepConv 梯度 |

## 11. 相关论文

- *Rep-Mamba: Re-Parameterization in Vision Mamba for Lightweight Remote Sensing Image Super-Resolution*（本仓库为该方法的 MindSpore 实现）
- *VDMamba: Vector Decomposition in Vision Mamba for Image Deraining and Beyond*, IEEE Transactions on Multimedia, 2026
- *PH-Mamba: Enhancing Mamba With Position Encoding and Harmonized Attention for Image Deraining and Beyond*, IEEE Transactions on Image Processing, 2026

## 12. 致谢

本工作得到人工智能与数字经济广东省实验室（深圳）开放课题基金（Open Research Fund from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), Grant GML-KF-24-09）资助。

## 13. 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可协议。
