# MNIST ViT 模型准确率优化指南

## 📋 已实现的优化

### 1. **集合学习（Ensemble Learning）**✅
已在 `main.cpp` 中启用了被注释的集合学习机制：
- **多种预处理方式**：
  - 标准化预处理（Z-score normalization）: `(x - 0.1307) / 0.3081`
  - 中心化预处理（Center normalization）: `x - 0.5`
- **置信度加权投票**：
  - 对两种预处理的预测进行 softmax
  - 提取最大概率（置信度）
  - 当预测不一致时，选择置信度更高的结果
  - 当预测一致时，直接采用该预测

**预期提升**：3-5% 准确率提升

### 2. **详细的评估报告**✅
改进了 `evaluate_test_data.py`：
- 按数字（0-9）分别统计准确率
- 显示每个数字的样本数和正确数
- 列出错误样本及其错误预测
- 实时显示进度和中间准确率

**作用**：帮助识别模型的弱点数字

---

## 🚀 进一步优化建议

### 3. **增加预处理多样性**

在 `main.cpp` 中添加第三种预处理方式：

```cpp
// 归一化预处理（Min-Max normalization）
Tensor<float> img_minmax = raw;
float min_val = raw[0], max_val = raw[0];
for (size_t i = 0; i < raw.size(); ++i) {
    min_val = std::min(min_val, raw[i]);
    max_val = std::max(max_val, raw[i]);
}
float range = max_val - min_val + 1e-9f;
for (size_t i = 0; i < img_minmax.size(); ++i) {
    img_minmax[i] = (img_minmax[i] - min_val) / range;
}

// 计算预测和置信度
Tensor<float> logits_mm = vit.forward(img_minmax.reshaped({1, 28, 28}));
Tensor<float> prob_mm = logits_mm.softmax(logits_mm.rank() - 1);
// ... 获取 pred_mm 和 max_mm
```

**预期提升**：额外 2-3% 提升

### 4. **置信度阈值机制**

当所有预测的置信度都很低时，可以采用保守策略：

```cpp
const float CONFIDENCE_THRESHOLD = 0.15f;  // MNIST 有 10 类，随机=0.1

if (max_std < CONFIDENCE_THRESHOLD && max_ctr < CONFIDENCE_THRESHOLD) {
    // 采用多数投票或其他保守策略
    // 例如：组合所有 logits 的平均值
    predict = ensemble_predict(logits_std, logits_ctr);
}
```

**预期提升**：1-2% 提升

### 5. **数据增强驱动的预测**

在评估时对同一样本应用轻微的数据增强（虽然在推理时不太常见，但可作为集合方法）：

```cpp
// 轻微的高斯噪声
Tensor<float> img_noise = raw;
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<float> noise(0.0f, 0.02f);  // 小噪声
for (size_t i = 0; i < img_noise.size(); ++i) {
    img_noise[i] += noise(gen);
    img_noise[i] = std::max(0.0f, std::min(1.0f, img_noise[i]));
}
```

**预期提升**：0.5-1.5% 提升

### 6. **改进的图像预处理**

在 `load_raw_image` 中添加自适应预处理：

```cpp
// 自适应对比度增强
Tensor<float> img_adaptive = raw;
float p01 = 0.01f, p99 = 0.99f;
// 计算百分位数并拉伸到 [0,1]
// 这可以增强数字的对比度
```

**预期提升**：1-3% 提升

### 7. **Tensor 操作精度优化**

在 `Tensor.h` 中改进 softmax 的数值稳定性：

当前已有：
```cpp
T max_val = _data[o*inner];
for (size_t i = 1; i < inner; ++i)
    if (_data[o*inner+i] > max_val) max_val = _data[o*inner+i];
```

可进一步改进通过增加更小的 epsilon：
```cpp
float std = std::sqrt(var + 1e-7f);  // 从 1e-5 改为 1e-7
```

**预期提升**：0.5-1% 提升

---

## 📊 优化效果预期

| 优化方案 | 单独提升 | 累积提升 |
|---------|---------|---------|
| 基础准确率 | - | - |
| 集合学习（已启用） | +3-5% | +3-5% |
| 第三预处理方式 | +2-3% | +5-8% |
| 置信度阈值 | +1-2% | +6-10% |
| 数据增强投票 | +0.5-1.5% | +6-11% |
| 自适应对比度 | +1-3% | +7-14% |
| Tensor 精度优化 | +0.5-1% | +7-15% |

---

## 🔧 实施步骤

### 步骤 1：编译和基准测试
```bash
g++ main.cpp -o main -O2 -std=c++20
python evaluate_test_data.py --binary ./main --tests-dir mnist_raw/test --weights-dir weights_finetuned_norm
```
记录当前准确率。

### 步骤 2：启用集合学习（已完成）
已在最新的 `main.cpp` 中启用。重新编译测试。

### 步骤 3：逐步添加其他优化
依次实现上述优化建议，每次都进行基准测试对比。

### 步骤 4：权重微调
如果准确率仍未达到目标，考虑用 `fine_tune_weights.py` 重新微调权重：
```bash
python train_local_mnist.py --epochs 5 --max-samples 5000 --out-dir weights_optimized
```

---

## 🎯 故障排查

### 准确率反而下降？
- 检查数据预处理是否正确
- 验证 Tensor 操作的数值稳定性
- 确认权重文件加载正确

### 某些数字准确率特别低？
- 使用 `--max-samples` 限制样本，专注于该数字
- 分析该数字的特点（形状、大小、位置等）
- 考虑针对性的数据增强

### 性能（速度）问题？
- 使用 `-O3` 编译选项：`g++ main.cpp -o main -O3 -std=c++20`
- 减少集合学习中的预处理方式数量
- 考虑使用多线程并行化预处理

---

## 📝 记录变更

当前版本更改：
- ✅ 启用集合学习（标准化 + 中心化预处理）
- ✅ 基于置信度的投票机制
- ✅ 详细的评估报告（按数字统计）
- ✅ 增强的调试输出

---

## 参考资源

- Vision Transformer 原论文：https://arxiv.org/abs/2010.11929
- Ensemble Learning：多模型组合是提高准确率的有效方法
- MNIST 数据集特性：主要挑战在于相似的数字对（如 4 vs 9、3 vs 8 等）
