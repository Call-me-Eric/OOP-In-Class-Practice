# ViT MNIST 手写数字推理项目

## 项目概述

本项目实现了一个基于 Tiny Vision Transformer 的 MNIST 手写数字分类推理系统。项目结构遵循实验要求，将任务拆分为：

- `Tensor.h`：张量类与基础张量运算
- `Model.h`：ViT 模型与 Transformer 模块实现
- `main.cpp`：推理入口、数据加载与权重注入

## 团队分工

- `Tensor.h`：@linnnn
- `Model.h`：@Call_me_Eric
- `main.cpp`：@bianbiandaren

## 目录结构

- `main.cpp`：推理程序入口
- `Model.h`：ViT 模型定义与模块组合
- `Tensor.h`：张量运算与算子实现
- `convert_image.py`：PNG → MNIST raw 二进制转换工具
- `evaluate_test_data.py`：批量评估脚本
- `fine_tune_weights.py`：PyTorch 微调现有权重并导出 `.wts` 格式
- `mnist_optimize.py`：MNIST 数据处理与优化辅助脚本
- `weights/`、`weights_finetuned_norm/`：预置模型权重
- `mnist_raw/test/`：raw 测试样本

## 编译与运行

编译：

```bash
g++ main.cpp -o main -O2 -std=c++20
```

运行：

```bash
./main <image_path> [weight_dir] [debug_flag]
```

参数说明：

- `image_path`：输入图片文件路径，支持 `*.raw` 原始字节文件和 `*.png` 图片文件
- `weight_dir`：权重目录（可选），默认 `weights_finetuned_norm`
- `debug_flag`：是否输出调试信息，`0` 为关闭，`1` 为开启，默认 `0`

示例：

```bash
./main mnist_raw/test/test_0000_label_1.raw weights_finetuned_norm 0
```

## 输入格式

- `*.raw`：28×28 灰度图像的裸二进制文件，784 字节，按行优先存储
- `*.png`：程序会调用 `convert_image.py` 自动转换为 `*.raw`，并读取转换结果

使用 `convert_image.py` 转换 PNG：

```bash
python convert_image.py
```

该脚本会执行：

1. 将图像转换为灰度
2. 缩放为 28×28
3. 输出 784 字节的 raw 文件

> 注意：脚本默认假设输入为黑底白字手写数字。若输入为白底黑字，需要解注释反色处理行。

## 权重格式

权重文件位于 `weight_dir`，每个 `.wts` 文本文件首行包含形状信息，后续行为空格分隔的浮点数。例如：

```text
# shape: 32 10
0.123 0.456 ...
```

`main.cpp` 会读取以下关键权重文件：

- `patch.weight.wts`
- `patch.bias.wts`
- `cls_token.wts`
- `pos_embed.wts`
- `blocks.0.*.wts`, `blocks.1.*.wts`
- `head.weight.wts`
- `head.bias.wts`

## 评估

使用 `evaluate_test_data.py` 对 `mnist_raw/test` 下的样本进行批量评估：

```bash
python evaluate_test_data.py --binary ./main --tests-dir mnist_raw/test --weights-dir weights_finetuned_norm
```

该脚本会自动读取标签并输出准确率。

## 已完成内容

- 完成核心模型结构定义与前向推理
- 完成张量运算库与广播、矩阵乘法、softmax 等基础算子
- 完成推理程序入口及多种输入处理流程
- 完成 PNG 转 RAW 和批量评估脚本
- 完成文档说明与使用说明
