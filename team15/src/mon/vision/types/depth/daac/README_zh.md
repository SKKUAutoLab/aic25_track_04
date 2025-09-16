<div align="center">
<h1>Depth Anything At Any Condition</h1>

[**孙博远**](https://bbbbchan.github.io)<sup>1*</sup> · [**金莫迪**](https://ghost233lism.github.io/)<sup>1*</sup> · [**尹博文**](https://yinbow.github.io/)<sup>1</sup> · [**侯淇彬**](https://houqb.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>南开大学 天津市视觉计算与智能感知重点实验室（VCIP）

*共同第一作者 &emsp;&dagger;通讯作者


**[English](README.md) | 简体中文**

<a href="#"><img src='https://img.shields.io/badge/Paper-Coming Soon-red' alt='Paper PDF'></a>
<a href="https://ghost233lism.github.io/depthanything-AC-page/ "><img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
<a href='https://huggingface.co/ghost233lism/DepthAnything-AC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/spaces/ghost233lism/DepthAnything-AC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange' alt='Demo'></a>
</div>

**DepthAnything-AC** 是一个基于 [DepthAnything-V2](https://github.com/DepthAnything/Depth-Anything-V2) 微调的鲁棒单目深度估计（MDE）模型，专为**在各种多样化和具有挑战性的环境条件下进行零样本深度估计**而设计，包括低光照、恶劣天气和传感器失真。

为了解决损坏场景中缺乏高质量标注的问题，我们引入了一个轻量级的**无监督一致性正则化**框架，使模型能够在无标签数据上进行训练。此外，我们提出的**空间距离约束**帮助模型学习几何关系，增强语义边界和精细细节。

![teaser](assets/teaser.png)


## 新闻

<div align="center">
<img src="assets/depthanything-AC-video.gif" alt="video" width="100%">
</div>

- **2025-07-02:** 🔥 DepthAnything AC代码库初始发布
- **2025-07-02:** 🔥 预训练模型和评估基准发布

## 模型架构

![architecture](assets/architecture.png)


## 安装

### 环境要求

- Python>=3.9
- torch==2.3.0
- torchvision==0.18.0
- torchaudio==2.3.0
- cuda==12.1

### 安装步骤

```bash
git clone https://github.com/your-repo/Depth-Anything-AC
cd Depth-Anything-AC
conda create -n depth_anything_ac python=3.9
conda activate depth_anything_ac
pip install -r requirements.txt
```


## 使用
### 获取 Depth-Anything-AC 模型
从huggingface下载预训练检查点：
```bash
mkdir checkpoints
cd checkpoints

# (可选) 使用huggingface镜像
export HF_ENDPOINT=https://hf-mirror.com

# 从huggingface下载DepthAnything-AC模型
huggingface-cli download --resume-download ghost233lism/DepthAnything-AC --local-dir ghost233lism/DepthAnything-AC
```

我们也在Google Drive提供DepthAnything-AC模型：[下载](https://drive.google.com/drive/folders/1yjM7_V9XQlL-taoRTbMq7aoCh1-Xr-ya?usp=sharing)

### 快速推理

我们在 `tools/` 中提供了单张/批量图像输入的快速推理脚本。请参考[推理](./tools/README.md)获取详细信息。

### 训练
我们提供DepthAnything-AC的完整训练过程，包括一致性正则化、空间距离提取/约束和广泛使用的仿射不变损失函数。

准备你在 `configs/` 中的配置文件并运行：

```bash
bash tools/train.sh <num_gpu> <port> <dataset>
```

### 评估
我们提供对DA-2K、增强DA-2K、KITTI、NYU-D、Sintel、ETH3D、DIODE、NuScenes-Night、RobotCar-night、DS-rain/cloud/fog、KITTI-C基准的直接评估。您可以参考 `configs/` 了解更多详情。

```bash
bash tools/val.sh <num_gpu> <port>
```

## 实验结果

### 定量结果

#### DA-2K多条件鲁棒性结果

在增强的多条件DA-2K基准上的定量结果，包括复杂光照和气候条件。评估指标为**准确率** ↑。

| 方法 | 编码器 | **DA-2K** | **DA-2K dark** | **DA-2K fog** | **DA-2K snow** | **DA-2K blur** |
|:-----|:-----:|:---------:|:---------------:|:--------------:|:---------------:|:---------------:|
| DynaDepth | ResNet | 0.655 | 0.652 | 0.613 | 0.605 | 0.633 |
| EC-Depth | ViT-S | 0.753 | 0.732 | 0.724 | 0.713 | 0.701 |
| STEPS | ResNet | 0.577 | 0.587 | 0.581 | 0.561 | 0.577 |
| RobustDepth | ViT-S | 0.724 | 0.716 | 0.686 | 0.668 | 0.680 |
| Weather-Depth | ViT-S | 0.745 | 0.724 | 0.716 | 0.697 | 0.666 |
| DepthPro | ViT-S | 0.947 | 0.872 | 0.902 | 0.793 | 0.772 |
| DepthAnything V1 | ViT-S | 0.884 | 0.859 | 0.836 | 0.880 | 0.821 |
| DepthAnything V2 | ViT-S | 0.952 | 0.910 | 0.922 | 0.880 | 0.862 |
| **DepthAnything-AC** | ViT-S | **0.953** | **0.923** | **0.929** | **0.892** | **0.880** |

#### 真实复杂基准上的零样本相对深度估计

在包括夜间场景、恶劣天气条件和复杂环境因素的挑战性真实世界场景上的零样本评估结果。所有结果使用ViT-S编码器。

| 方法 | 编码器 | **NuScenes-夜间** | | **RobotCar-夜间** | | **DS-雨** | | **DS-云** | | **DS-雾** | |
|:-----|:-----:|:----------------:|:---:|:----------------:|:---:|:---------:|:---:|:----------:|:---:|:--------:|:---:|
| | | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ |
| DynaDepth | ResNet | 0.381 | 0.394 | 0.512 | 0.294 | 0.239 | 0.606 | 0.172 | 0.608 | 0.144 | 0.901 |
| EC-Depth | ViT-S | 0.243 | 0.623 | 0.228 | 0.552 | 0.155 | 0.766 | 0.158 | 0.767 | 0.109 | 0.861 |
| STEPS | ResNet | 0.252 | 0.588 | 0.350 | 0.367 | 0.301 | 0.480 | 0.252 | 0.588 | 0.216 | 0.641 |
| RobustDepth | ViT-S | 0.260 | 0.597 | 0.311 | 0.521 | 0.167 | 0.755 | 0.168 | 0.775 | 0.105 | 0.882 |
| Weather-Depth | ViT-S | - | - | - | - | 0.158 | 0.764 | 0.160 | 0.767 | 0.105 | 0.879 |
| Syn2Real | ViT-S | - | - | - | - | 0.171 | 0.729 | - | - | 0.128 | 0.845 |
| DepthPro | ViT-S | 0.218 | 0.669 | 0.237 | 0.534 | **0.124** | **0.841** | 0.158 | 0.779 | **0.102** | **0.892** |
| DepthAnything V1 | ViT-S | 0.232 | 0.679 | 0.239 | 0.518 | 0.133 | 0.819 | 0.150 | **0.801** | 0.098 | 0.891 |
| DepthAnything V2 | ViT-S | 0.200 | 0.725 | 0.239 | 0.518 | 0.125 | 0.840 | 0.151 | 0.798 | 0.103 | 0.890 |
| **DepthAnything-AC** | ViT-S | **0.198** | **0.727** | **0.227** | **0.555** | 0.125 | 0.840 | **0.149** | **0.801** | 0.103 | 0.889 |

*粗体：最佳性能，下划线：第二佳性能。NuScenes-夜间和RobotCar-夜间代表夜间驾驶场景。DS-雨、DS-云和DS-雾是DrivingStereo数据集。*

#### 合成KITTI-C基准上的零样本相对深度估计

在合成KITTI-C损坏基准上的零样本评估结果，测试对各种图像退化和损坏的鲁棒性。

| 方法 | 编码器 | **暗光** | | **雪** | | **运动模糊** | | **高斯模糊** | |
|:-----|:-----:|:--------:|:---:|:--------:|:---:|:----------:|:---:|:------------:|:---:|
| | | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ |
| DynaDepth | ResNet | 0.163 | 0.752 | 0.338 | 0.393 | 0.234 | 0.609 | 0.274 | 0.501 |
| STEPS | ResNet | 0.230 | 0.631 | 0.242 | 0.622 | 0.291 | 0.508 | 0.204 | 0.692 |
| DepthPro | ViT-S | 0.145 | 0.793 | 0.197 | 0.685 | 0.170 | 0.746 | 0.170 | 0.745 |
| DepthAnything V2 | ViT-S | **0.130** | 0.832 | 0.115 | 0.872 | 0.127 | 0.840 | 0.157 | 0.785 |
| **DepthAnything-AC** | ViT-S | **0.130** | **0.834** | **0.114** | **0.873** | **0.126** | **0.841** | **0.153** | **0.793** |

*KITTI-C包含合成损坏：暗光（低光照条件）、雪（天气模拟）、运动（运动模糊）和高斯（噪声损坏）。*

## 引用

如果您发现这项工作有用，请考虑引用：

```bibtex
@article{depth_anything_ac,
  title={Depth Anything at Any Condition},
  author={Sun, Boyuan and Modi Jin and Bowen Yin and Hou, Qibin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 许可证

本代码采用[知识共享署名-非商业性使用 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc/4.0/)，仅限非商业用途。
请注意，任何商业使用本代码都需要在使用前获得正式许可。

## 联系方式

如有技术问题，请联系
sbysbysby123[AT]gmail.com 或 jin_modi[AT]mail.nankai.edu.cn

如需商业许可，请联系 andrewhoux[AT]gmail.com。

## 致谢

我们感谢[DepthAnything](https://github.com/LiheYoung/Depth-Anything)和[DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2)作者的基础性工作。我们也感谢[DINOv2](https://github.com/facebookresearch/dinov2)提供的强大视觉编码器，[CorrMatch](https://github.com/BBBBchan/CorrMatch)提供的代码库，以及[RoboDepth](https://github.com/ldkong1205/RoboDepth)的贡献。 
