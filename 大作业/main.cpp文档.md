# main.cpp 文档

## 概述

`main.cpp` 是 Tiny Vision Transformer 推理系统的入口文件，负责：

1. 从 `*.raw` 或 `*.png` 文件中读取 MNIST 灰度图像
2. 对输入图像做标准化预处理
3. 从权重目录加载模型参数
4. 构建 `VisionTransformer`
5. 执行前向推理，输出预测分类结果

---

## 命令行接口

```bash
./main <image_path> [weight_dir] [debug_flag]
```

| 参数 | 说明 |
|------|------|
| `image_path` | 输入图像路径，支持 `*.raw` 或 `*.png` |
| `weight_dir` | 权重目录，默认 `weights_finetuned_norm` |
| `debug_flag` | 0/1，是否输出调试日志，默认 0 |

---

## 模型超参数

以下参数在 `main` 函数中以常量形式硬编码：

| 参数 | 值 | 说明 |
|------|----|------|
| `INPUT_H / INPUT_W` | 28 | 输入图像尺寸 |
| `PATCH_H / PATCH_W` | 7 | Patch 大小，生成 16 个 patch |
| `HIDDEN_DIM` | 32 | Token 隐藏维度 |
| `NUM_HEADS` | 4 | 多头注意力头数 |
| `MLP_DIM` | 64 | 前馈网络隐藏维度 |
| `NUM_LAYERS` | 2 | Transformer 编码器块数量 |
| `NUM_CLASSES` | 10 | 分类类别数 |

---

## 功能函数说明

### `load_tensor`

```cpp
Tensor<float> load_tensor(const string& filepath);
```

从 `.wts` 文本文件读取张量数据。

**文件格式要求**：

- 第一行格式为 `# shape: R C` 或 `# shape: N`
- 后续行为空格分隔的浮点数数据

**处理规则**：

- `R C` 解析为形状 `{R, C}`
- `N` 解析为 1D bias，自动转换为 `{1, N}`
- 如果数据数量和张量容量不匹配，程序会输出警告，但仍会按最小值填充

**异常**：
- 文件打开失败
- shape 行格式非法

---

### `load_image`

`main.cpp` 中有两种图像读取方式：

- `load_image`：包含 PNG 转 RAW 的自动转换逻辑
- `load_raw_image`：直接从 raw 文件读取像素数据

`load_image` 会在输入文件名为 `*.png` 时，调用 `convert_image.py` 产生临时 raw 文件；如果转换成功，则读取该 raw 文件，并在读取后删除临时文件。

读取后的原始像素值会除以 255.0f，得到 `[0, 1]` 的浮点张量。

---

### 标准化预处理

程序将读取的 raw 图像分别进行两类预处理：

1. `img_std`：使用 MNIST 常用均值 `0.1307f` 和标准差 `0.3081f` 进行标准化
2. `img_center`：直接减去 `0.5f`，得到中心化结果

然后对两种预处理结果分别执行一次前向推理，最终根据 softmax 置信度选择更可靠的预测值。

---

### `load_block`

```cpp
TransformerBlock load_block(const string& wdir, int idx,
                            size_t hidden_dim, size_t num_heads, size_t mlp_dim);
```

从权重目录中加载第 `idx` 个 Transformer 编码器块参数，并返回已初始化的 `TransformerBlock`。

加载文件包括：

- `blocks.<idx>.attn.q.weight.wts`
- `blocks.<idx>.attn.q.bias.wts`
- `blocks.<idx>.attn.k.weight.wts`
- `blocks.<idx>.attn.k.bias.wts`
- `blocks.<idx>.attn.v.weight.wts`
- `blocks.<idx>.attn.v.bias.wts`
- `blocks.<idx>.attn.o.weight.wts`
- `blocks.<idx>.attn.o.bias.wts`
- `blocks.<idx>.mlp.fc1.weight.wts`
- `blocks.<idx>.mlp.fc1.bias.wts`
- `blocks.<idx>.mlp.fc2.weight.wts`
- `blocks.<idx>.mlp.fc2.bias.wts`
- `blocks.<idx>.norm1.gamma.wts`
- `blocks.<idx>.norm1.beta.wts`
- `blocks.<idx>.norm2.gamma.wts`
- `blocks.<idx>.norm2.beta.wts`

---

## main() 运行流程

1. 解析命令行参数，默认权重目录 `weights_finetuned_norm`
2. 加载原始 raw 图像到 `{1, 28, 28}` 张量
3. 生成 `img_std` 与 `img_center` 两种预处理输入
4. 读取 patch 投影权重和位置编码：
   - `patch.weight.wts`
   - `patch.bias.wts`
   - `cls_token.wts`
   - `pos_embed.wts`
5. 构建 `PatchEmbedding` 和 `VisionTransformer`
6. 读取并设置 `NUM_LAYERS` 个 `TransformerBlock`
7. 读取分类头权重 `head.weight.wts` 与 `head.bias.wts`
8. 对 `img_std` 和 `img_center` 分别推理
9. 依据 softmax 置信度选取最终预测结果
10. 输出预测数字到标准输出

---

## 权重文件要求

- `patch.weight.wts`
- `patch.bias.wts`
- `cls_token.wts`
- `pos_embed.wts`
- `blocks.0.*.wts`
- `blocks.1.*.wts`
- `head.weight.wts`
- `head.bias.wts`

其中 `pos_embed.wts` 需要 reshape 为 `{1, 17, 32}`。

---

## 输出行为

- 程序最终只会在 `stdout` 输出一个整数预测结果
- `debug_flag=1` 时，会将详细推理日志输出到 `stderr`

---

## 图像输入约定

本项目所用输入图像均为**黑底白字**的手写数字图片（与 MNIST 数据集风格一致）。

在传入 `vit_infer.exe` 之前，需先使用 `convert_image.py` 将 PNG 图像转换为程序可读的裸二进制格式：

```bash
./main mnist_raw/test/test_0000_label_1.raw weights_finetuned_norm 1
```

若输入为 PNG：

```bash
./main test_image.png weights_finetuned_norm 0
```

程序会自动调用 `convert_image.py` 并生成临时 raw 文件。

`convert_image.py` 的转换逻辑如下：

```python
from PIL import Image
import numpy as np

def png_to_raw(input_png, output_raw):
    img = Image.open(input_png)
    img = img.convert("L")      # 转为灰度图
    img = img.resize((28, 28))  # 缩放至 28×28
    img_array = np.array(img, dtype=np.uint8)
    # 黑底白字无需反色，若为白底黑字请取消下一行注释
    # img_array = 255 - img_array
    img_array.tofile(output_raw)
    print(f"转换完成")

if __name__ == "__main__":
    input_png = input("请输入 PNG 文件路径: ")
    output_raw = input("请输入输出 RAW 文件路径: ")
    png_to_raw(input_png, output_raw)
```

**注意：**

- 输入图片必须为**黑底白字**；若为白底黑字，需在 `convert_image.py` 中启用 `img_array = 255 - img_array` 反色处理
- `convert_image.py` 会自动将图片缩放至 28×28 并转为灰度图，无需手动预处理
- 输出的 `.raw` 文件为 784 字节的裸二进制文件，不含任何文件头，直接按行优先顺序存储像素值

---

## 注意事项

- 所有推理日志输出到 `stderr`，便于与标准输出分离
- `load_tensor` 对数据量不匹配的情况只发出警告，不终止程序；调用方应确保权重文件与模型结构严格对应
- 本文件未实现 PDF 要求的 `export_attention` 功能（导出指定层指定头的 attention 矩阵），如需此功能可通过 `TransformerBlock` 内部的 `MultiHeadAttention::get_attention_map()` 接口扩展实现