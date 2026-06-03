# Model.h 文档

本文件说明 `大作业/Model.h` 中已实现的类、输入输出形状、参数注入接口以及使用方法。

## 约定

- `Tensor<T>::shape()` 返回张量形状，例如 `{B, T, hidden_dim}`。
- `Tensor<T>` 的元素访问使用 `operator()` 多维索引。
- `Tensor.h` 已实现 `matmul`, `bias_add`, `softmax`, `slice`, `concat`, `reshape`, `permute` 等操作。
- `Model.h` 仅定义网络结构与前向计算，权重、偏置、CLS token、位置编码等参数由外部加载后注入。

## Layer 类

`Layer` 是模型中所有层的抽象基类。

```cpp
class Layer {
public:
    virtual Tensor<float> forward(const Tensor<float>& x) = 0;
    virtual ~Layer() = default;
};
```

## Linear 类

`Linear` 表示线性变换层，用于实现 `xW + b`。

### 构造方法

- `Linear(size_t in_f = 0, size_t out_f = 0)`
  - 以参数维度构造权重和偏置张量
- `Linear(const Tensor<float>& weight, const Tensor<float>& bias)`
  - 直接使用外部已加载的权重与偏置

### 关键接口

- `set_weights(const Tensor<float>& weight, const Tensor<float>& bias)`
- `const Tensor<float>& weight() const`
- `const Tensor<float>& bias() const`

### forward

- 输入形状：`{B, tokens, in_f}`
- 输出形状：`{B, tokens, out_f}`
- 计算流程：
  1. `x.matmul(_weight)`
  2. `bias_add(_bias)`

## PatchEmbedding 类

`PatchEmbedding` 将输入图像切分为 patch，并映射到 token 向量。

### 构造方法

- `PatchEmbedding(size_t input_h = 28, size_t input_w = 28, size_t patch_h = 7, size_t patch_w = 7, size_t hidden_dim = 32)`
  - 构造 `cls_token`、`pos_embedding` 和 `linear` 参数容器
- `PatchEmbedding(const Tensor<float>& cls_token, const Tensor<float>& pos_embedding, const Linear& linear, size_t input_h = 28, size_t input_w = 28, size_t patch_h = 7, size_t patch_w = 7)`
  - 使用外部加载的参数直接构造对象

### 参数注入

- `set_cls_token(const Tensor<float>& token)`
- `set_pos_embedding(const Tensor<float>& embedding)`
- `set_projection(const Linear& proj)`

### forward

- 输入形状：`{B, input_h, input_w}`
- 输出形状：`{B, 1 + num_patches, hidden_dim}`

计算流程：

1. 将图像划分为 `patch_h x patch_w` 小块
2. 每块 reshape 为 `{B, patch_h * patch_w}`
3. 线性映射到 `{B, hidden_dim}`
4. 添加对应的位置信息
5. 拼接 CLS token 与 patch tokens

> 该实现会将 `cls_token` 与 `pos_embedding.slice(1, 0, 1)` 相加，保证 CLS token 也携带位置编码。

## LayerNorm 类

`LayerNorm` 实现了基于最后一维的归一化，并支持 gamma/beta 可学习参数。

### 构造方法

- `LayerNorm()`
- `LayerNorm(const Tensor<float>& gamma, const Tensor<float>& beta)`

### 参数注入

- `set_weights(const Tensor<float>& gamma, const Tensor<float>& beta)`

### forward

- 输入输出形状一致
- 对每个样本最后一维计算均值与方差
- 执行标准化后，若存在 `_gamma` 和 `_beta`，则做缩放平移

## MultiHeadAttention 类

`MultiHeadAttention` 实现标准多头自注意力。

### 构造方法

- `MultiHeadAttention(size_t hidden_dim, size_t num_heads)`
  - 计算 `head_dim = hidden_dim / num_heads`
  - 需要 `hidden_dim % num_heads == 0`

### 参数注入

- `set_q_proj(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_k_proj(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_v_proj(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_out_proj(const Tensor<float>& weight, const Tensor<float>& bias)`

### forward

- 输入形状：`{B, T, hidden_dim}`
- 计算流程：
  1. 生成 `Q`, `K`, `V`
  2. reshape 为 `{B, T, num_heads, head_dim}`
  3. permute 为 `{B, num_heads, T, head_dim}`
  4. 计算 `scores = Q * K^T`
  5. 缩放 `scores / sqrt(head_dim)`
  6. 对最后一维执行 `softmax`
  7. 计算注意力加权输出
  8. reshape 回 `{B, T, hidden_dim}`
  9. 通过 `_out_proj` 输出最终结果

### 注意力地图

- `get_attention_map()` 返回最后一次计算的注意力权重，形状为 `{B, num_heads, T, T}`

## MLP 类

`MLP` 实现 Transformer 中的前馈网络。

### 构造方法

- `MLP(size_t hidden_dim = 32, size_t mlp_dim = 64)`

### 参数注入

- `set_fc1(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_fc2(const Tensor<float>& weight, const Tensor<float>& bias)`

### forward

- 先执行 `_fc1`
- 再执行 `GELU`
- 最后执行 `_fc2`

## TransformerBlock 类

`TransformerBlock` 组合了 LayerNorm、MultiHeadAttention 和 MLP。

### 构造方法

- `TransformerBlock(size_t hidden_dim = 32, size_t num_heads = 4, size_t mlp_dim = 64)`

### 参数注入

- `set_attention_weights(...)`
- `set_mlp_weights(...)`
- `set_norm_weights(...)`

### forward

1. 先对输入执行 `_norm1`
2. 进行自注意力，得到 `attn_output`
3. 残差相加 `residual1 = x + attn_output`
4. 对 `residual1` 执行 `_norm2`
5. 经过 MLP 并与 `residual1` 相加

输出形状为 `{B, T, hidden_dim}`。

## VisionTransformer 类

`VisionTransformer` 实现完整的 ViT 推理前向流程。

### 构造方法

- `VisionTransformer(size_t input_h = 28, size_t input_w = 28, size_t patch_h = 7, size_t patch_w = 7, size_t hidden_dim = 32, size_t num_heads = 4, size_t mlp_dim = 64, size_t num_layers = 1, size_t num_classes = 10)`

### 参数注入

- `set_patch_embedding(const PatchEmbedding& patch_embed)`
- `set_block(size_t index, const TransformerBlock& block)`
- `set_head(const Tensor<float>& weight, const Tensor<float>& bias)`

### forward

- 先通过 `PatchEmbedding` 生成 token 序列
- 依次通过每个 `TransformerBlock`
- 对输出进行 `LayerNorm`
- 提取 CLS token 并通过分类头输出 logits

## 使用示例

```cpp
PatchEmbedding patch_embed(28, 28, 7, 7, 32);
VisionTransformer vit(28, 28, 7, 7, 32, 4, 64, 2, 10);

Tensor<float> cls_token({1, 1, 32});
Tensor<float> pos_embed({1, 17, 32});
Linear patch_proj(49, 32);
patch_embed.set_cls_token(cls_token);
patch_embed.set_pos_embedding(pos_embed);
patch_embed.set_projection(patch_proj);

vit.set_patch_embedding(patch_embed);

TransformerBlock block0(32, 4, 64);
TransformerBlock block1(32, 4, 64);
// 注入 block0、block1 参数
vit.set_block(0, block0);
vit.set_block(1, block1);

Tensor<float> head_w({32, 10});
Tensor<float> head_b({1, 10});
vit.set_head(head_w, head_b);

Tensor<float> image({1, 28, 28});
Tensor<float> logits = vit.forward(image);
```

## 已完成内容总结

- `Linear`：实现仿射变换与外部权重注入
- `PatchEmbedding`：实现 patch 划分、CLS token 追加、位置编码注入
- `LayerNorm`：实现基于最后一维的归一化并支持 gamma/beta
- `MultiHeadAttention`：实现 Q/K/V 生成、缩放 softmax、注意力加权与输出映射
- `MLP`：实现前馈网络与 GELU 激活
- `TransformerBlock`：实现残差连接与标准 Transformer 编码器结构
- `VisionTransformer`：实现完整 ViT 前向推理框架

### 参数注入

- `set_patch_embedding(const PatchEmbedding& patch_embed)`
- `set_head(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_block(size_t index, const TransformerBlock& block)`

### forward 方法

- 先通过 `PatchEmbedding` 获取 token 序列。
- 再依次通过多个 `TransformerBlock`。
- 最后对 token 做 `LayerNorm` 并提取 `CLS` token。
- 输出 `Linear` 分类头结果，形状为 `{B, num_classes}`。

## 参数加载与使用示例

以下示例展示如何在外部加载参数后使用 `Model.h` 类。

```cpp
// 1. 创建网络结构，超参数由代码给出
PatchEmbedding patch_embed(28, 28, 7, 7, 32);
MultiHeadAttention mha(32, 4);
MLP mlp(32, 64);
TransformerBlock block(32, 4, 64);
VisionTransformer vit(28, 28, 7, 7, 32, 4, 64, 1, 10);

// 2. 从文件加载权重和偏置，假设已经得到以下 Tensor
Tensor<float> q_weight({32, 32});
Tensor<float> q_bias({1, 32});
Tensor<float> k_weight({32, 32});
Tensor<float> k_bias({1, 32});
Tensor<float> v_weight({32, 32});
Tensor<float> v_bias({1, 32});
Tensor<float> out_weight({32, 32});
Tensor<float> out_bias({1, 32});

// 3. 注入模型参数
mha.set_q_proj(q_weight, q_bias);
mha.set_k_proj(k_weight, k_bias);
mha.set_v_proj(v_weight, v_bias);
mha.set_out_proj(out_weight, out_bias);

Tensor<float> fc1_weight({32, 64});
Tensor<float> fc1_bias({1, 64});
Tensor<float> fc2_weight({64, 32});
Tensor<float> fc2_bias({1, 32});
block.set_mlp_weights(fc1_weight, fc1_bias, fc2_weight, fc2_bias);

Tensor<float> cls_token({1, 1, 32});
Tensor<float> pos_embed({1, 17, 32});
Linear project(49, 32);
patch_embed.set_cls_token(cls_token);
patch_embed.set_pos_embedding(pos_embed);
patch_embed.set_projection(project);

Tensor<float> head_weight({32, 10});
Tensor<float> head_bias({1, 10});
vit.set_head(head_weight, head_bias);

// 4. 前向推理
Tensor<float> image({1, 28, 28});
Tensor<float> logits = vit.forward(image);
```

> 说明：本示例中权重加载过程不在 `Model.h` 内实现，调用方需要根据文件格式自行读取 Tensor 数据后调用 setter 注入。

## 已完成内容总结

- `Layer`：定义 `forward` 抽象接口。
- `Linear`：实现 `matmul` + `bias_add` 的仿射变换，并支持外部参数注入。
- `PatchEmbedding`：实现结构超参数接口，并支持外部注入 `cls_token`、`pos_embedding`、`projection`。
- `LayerNorm`：实现最后一维 softmax 归一化。
- `MultiHeadAttention`：实现 Q/K/V 投影、注意力分数计算、缩放 softmax、注意力加权求和与输出投影，并支持外部投影参数注入。
- `MLP`：实现前馈网络并支持外部参数注入。
- `TransformerBlock`：实现编码器块并支持外部注入注意力和 MLP 参数。
- `VisionTransformer`：实现 ViT 结构并支持外部注入分类头与子模块参数。
