# Tools 使用说明

[English](README.md) | **中文**

本目录包含了DepthAnything-AC项目的工具脚本。

## 推理脚本 (infer.py)

推理脚本支持图像和视频处理，具有灵活的批量处理功能。

### 基本使用

#### 单张图像推理
```bash
# 基本用法
python tools/infer.py --input image.jpg --output depth.png

# 指定模型和编码器
python tools/infer.py --input image.jpg --output depth.png --model checkpoints/depth_anything_AC_vits.pth --encoder vits

# 使用不同的颜色映射
python tools/infer.py --input image.jpg --output depth.png --colormap inferno
```

#### 单个视频推理
```bash
# 基本视频处理
python tools/infer.py --input video.mp4 --output depth_video.mp4

# 指定输出帧率
python tools/infer.py --input video.mp4 --output depth_video.mp4 --fps 30

# 为视频使用不同的颜色映射
python tools/infer.py --input video.mp4 --output depth_video.mp4 --colormap spectral
```

#### 批量处理
```bash
# 只处理目录中的图像
python tools/infer.py --input images/ --output output/ --mode images

# 只处理目录中的视频
python tools/infer.py --input videos/ --output output/ --mode videos

# 同时处理图像和视频（默认模式）
python tools/infer.py --input media/ --output output/ --mode mixed

# 递归处理子目录
python tools/infer.py --input dataset/ --output results/ --recursive
```

### 参数说明

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | `-i` | str | - | 输入图像/视频路径或目录 (必需) |
| `--output` | `-o` | str | - | 输出路径（文件或目录）(必需) |
| `--model` | `-m` | str | `checkpoints/depth_anything_v2_vits.pth` | 模型权重路径 |
| `--encoder` | - | str | `vits` | 编码器类型 (`vits`, `vitb`, `vitl`) |
| `--colormap` | - | str | `spectral` | 颜色映射 (`inferno`, `spectral`, `gray`) |
| `--depth_cap` | - | float | `80` | 深度值上限 |
| `--fps` | - | float | `None` | 输出视频帧率（默认使用输入视频帧率） |
| `--recursive` | `-r` | flag | `False` | 递归搜索子目录 |
| `--mode` | - | str | `mixed` | 处理模式 (`images`, `videos`, `mixed`) |

### 支持的文件格式

**图像**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

**视频**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`

### 使用示例

#### 示例1: 处理单张图像
```bash
python tools/infer.py -i test_image.jpg -o result.png
```
输出：
- `result.png` - 彩色深度图
- `result_raw.npy` - 原始深度数据

#### 示例2: 处理单个视频
```bash
python tools/infer.py -i test_video.mp4 -o depth_video.mp4 --fps 30
```
输出：
- `depth_video.mp4` - 带深度可视化的视频

#### 示例3: 仅批量处理图像
```bash
python tools/infer.py -i photos/ -o depth_results/ --mode images --recursive
```
递归处理 `photos/` 目录及其子目录中的所有图像，在 `depth_results/` 目录生成对应的深度图。

#### 示例4: 仅批量处理视频
```bash
python tools/infer.py -i videos/ -o depth_videos/ --mode videos --fps 25
```
处理 `videos/` 目录中的所有视频，生成25帧率的深度视频。

#### 示例5: 混合批量处理
```bash
python tools/infer.py -i media/ -o output/ --mode mixed --colormap inferno
```
处理 `media/` 目录中的图像和视频，使用inferno颜色映射进行可视化。

### 输出结构

处理目录时，输出会保持与输入相同的文件夹结构：
- 图像：`输入名称_depth.png` + `输入名称_depth_raw.npy`
- 视频：`输入名称_depth.mp4`

## 训练脚本

### train.sh
用于启动分布式训练的脚本。

```bash
bash tools/train.sh [GPU数量] [端口号]
```

### val.sh  
用于启动分布式评估的脚本。

```bash
bash tools/val.sh [GPU数量] [端口号] [数据集]
```

---

更多详细信息请参考项目主[README](../README_zh.md)文档。 
