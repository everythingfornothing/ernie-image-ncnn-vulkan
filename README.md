# ERNIE-Image-Turbo ncnn

这是 [`baidu/ERNIE-Image-Turbo`](https://huggingface.co/baidu/ERNIE-Image-Turbo)
的实验性 C++/ncnn 移植，用于在 Linux 本地完成文生图推理。运行时同时支持
ncnn CPU 与 Vulkan 后端；下载转换后的模型文件后，推理过程不依赖 Python。

本仓库包含 C++ 推理代码以及验收时实际使用的 ncnn 源码。FP32 部署模型约为
43 GiB，因此转换后的权重通过独立的 Hugging Face 仓库发布，不存放在本代码
仓库中。

## 当前状态

已经跑通并验证的推理链路为：

```text
UTF-8 Prompt
  -> C++ Tokenizer
  -> ncnn Text Encoder
  -> 动态文本长度、RoPE 和 Attention Mask
  -> 8 x（ncnn DiT + C++ FlowMatch Scheduler）
  -> ncnn VAE Decoder
  -> 1024 x 1024 RGB PNG
```

当前验收范围：

- 模型：ERNIE-Image-Turbo，关闭 Prompt Enhancer
- Batch size：1
- 分辨率：1024 x 1024
- 推理步数：8
- Guidance scale：1.0
- 运行精度：FP32
- 后端：ncnn CPU、ncnn Vulkan
- 动态 Prompt：CPU 已验证 N=3、14、30、64、256、1024；Vulkan 端到端已验证
  N=14 和 N=30
- 验证平台：Ubuntu 22.04、GCC 11.4、RTX 4080 SUPER、NVIDIA Driver
  580.105.08

当前版本以正确性验证为目标。低精度 Vulkan、可变图像分辨率、batch > 1、
Prompt Enhancer 和性能优化暂未纳入已验证范围。

## 仓库结构

```text
.
├── CMakeLists.txt
├── cpp/
│   ├── include/ernie_image/
│   ├── src/
│   └── tools/ernie_image_cli.cpp
└── third_party/
    ├── ncnn/
    └── tokenizers-cpp/
```

仓库内的 ncnn 基于 tag `20260526`、commit
`e54f7b1f88434e1d844ea0551b880a1cfb079ce1`，并包含本项目 Vulkan 验收所使用的
exact GELU 与 SDPA `[1,S]` mask 广播修正。复现当前结果时不要在未验证的情况下
替换为其他 ncnn 版本。

## 环境依赖

Ubuntu 22.04 安装命令：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build libpng-dev libvulkan-dev vulkan-tools \
  curl ca-certificates
```

`tokenizers-cpp` 的构建需要当前稳定版 Rust。Ubuntu 22.04 软件源中的 Rust 版本
可能过旧，建议通过 [rustup](https://rustup.rs/) 安装 stable toolchain，并确认
`cargo --version` 与 `rustc --version` 均可正常执行。首次编译时 Cargo 会下载
Rust crate 依赖。

如需使用 Vulkan，请先安装可正常工作的 GPU 驱动，并确认以下命令能够识别目标
GPU：

```bash
vulkaninfo --summary
```

构建过程使用仓库内置的 glslang 源码，不要求单独安装 glslang 软件包。

## 下载转换后的权重

转换后的 FP32 权重发布在
[`Coderdw/ernie-image-ncnn-vulkan`](https://huggingface.co/Coderdw/ernie-image-ncnn-vulkan)：

```bash
python3 -m pip install -U huggingface_hub
export HF_REPO_ID="Coderdw/ernie-image-ncnn-vulkan"
hf download "$HF_REPO_ID" --local-dir models/ernie-image-turbo
```

下载目录中必须直接包含 `tokenizer/`、`text_encoder/`、`dit/`、`vae/` 和
`rng/`，不能再额外嵌套一层目录。

## 编译

克隆代码仓库：

```bash
git clone https://github.com/everythingfornothing/ernie-image-ncnn-vulkan.git
cd ernie-image-ncnn-vulkan
```

编译 Vulkan 版本：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DERNIE_IMAGE_VULKAN=ON
cmake --build build --parallel
```

只编译 CPU 版本：

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DERNIE_IMAGE_VULKAN=OFF
cmake --build build-cpu --parallel
```

## 运行

检查模型目录和 Vulkan 设备：

```bash
./build/ernie_image_cli \
  --model-dir models/ernie-image-turbo \
  --device vulkan \
  --gpu-index 0 \
  --threads 16
```

只执行 Tokenizer 和 Text Encoder：

```bash
./build/ernie_image_cli \
  --model-dir models/ernie-image-turbo \
  --device vulkan \
  --encode '一只黑白相间的中华田园犬'
```

使用官方 seed=42 回归 latent 生成图片：

```bash
mkdir -p outputs
./build/ernie_image_cli \
  --model-dir models/ernie-image-turbo \
  --device vulkan \
  --gpu-index 0 \
  --threads 16 \
  --rng-mode reference \
  --seed 42 \
  --prompt '一只黑白相间的中华田园犬' \
  --output outputs/seed42.png
```

使用任意可复现随机种子生成图片：

```bash
./build/ernie_image_cli \
  --model-dir models/ernie-image-turbo \
  --device vulkan \
  --rng-mode portable \
  --seed 123 \
  --prompt '一只黑白相间的中华田园犬在草地上奔跑' \
  --output outputs/seed123.png
```

将 `--device vulkan` 改为 `--device cpu` 即可使用 CPU 后端。

`reference` 随机数模式只接受 seed=42，用于保持官方回归输入一致。其他 seed 必须
使用 `portable` 模式。`portable` 模式能够跨运行复现，但不保证与 PyTorch CUDA
随机数流逐字节一致。

## 已验证结果

| 案例 | 完成步数 | 总耗时 | CPU/Vulkan 图片 PSNR | 输出 |
|---|---:|---:|---:|---|
| N=14、seed=42、Vulkan | 8/8 | 408.27 s | 57.16 dB | 1024 x 1024 RGB |
| N=30、seed=42、Vulkan | 8/8 | 424.25 s | 64.43 dB | 1024 x 1024 RGB |

N=14 测试中，通过设备级采样得到的 GPU 峰值显存为 10,855 MiB。以上数据是
正确性优先的基线，不代表完成性能优化后的结果。

## 模型与第三方声明

ERNIE-Image-Turbo 由百度开发，上游模型仓库以 Apache-2.0 许可证发布。转换后的
权重仓库中保留了上游许可证。ncnn 与 tokenizers-cpp 的许可证分别保留在
`third_party/` 对应目录中。

本项目不是百度或腾讯 ncnn 的官方发布。项目所有者在公开发布前仍需为本项目
自行编写的 C++ 代码确定许可证。
