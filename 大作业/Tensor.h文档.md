# Tensor.h 文档

`Tensor.h` 是本项目的基础张量实现文件，提供多维张量存储、索引、形状变换和常用算子。

## 内部实现

### 数据存储

- `_data`: 存储张量元素的连续一维数组
- `_shape`: 存储各维度大小
- `_strides`: 存储每个维度对应的一维偏移步长

### 关键函数

- `compute_strides()`: 根据形状计算步长
- `to_flat_idx(...)`: 将多维索引转换为一维索引
- `count_elements(shape)`: 统计形状对应的元素总数

这些函数保证了 reshape、slice、permute 等操作的数据访问正确性。

## 公开接口

### 构造函数

- `Tensor()`：默认构造
- `Tensor(const vector<size_t>& shape)`：按形状分配内存
- `Tensor(const vector<size_t>& shape, const T& init_val)`：按形状分配并初始化

### 基础信息

- `size()`：总元素数
- `rank()`：维度数
- `shape()`：返回形状向量

### 访问与索引

- `operator[](size_t idx)`：一维索引访问
- `operator()(Index... idx)`：多维索引访问
- `print_info()`：输出形状与大小信息

## 形状变换

### reshape / reshaped

- `reshape(const vector<size_t>& new_shape)`：原地改变形状
- `reshaped(const vector<size_t>& new_shape)`：返回新形状张量

两者要求总元素数不变。

### permute

- `permute(const vector<size_t>& dims)`：维度重排
- `transpose()`：二维张量转置

### unsqueeze / squeeze

- `unsqueeze(size_t dim)`：在指定位置插入长度为 1 的维度
- `squeeze(size_t dim)`：去除指定的长度为 1 的维度

### slice

- `slice(size_t dim, size_t begin, size_t end)`：沿指定维度截取子张量

### concat

- `concat(const vector<Tensor<T>>& tensors, size_t dim)`：沿指定维度拼接张量

## 运算算子

### matmul

- `Tensor<T> matmul(const Tensor<T>& other) const`
- 支持批量矩阵乘法
- 要求两个输入至少为二维张量，且 `K` 维度匹配

### bias_add

- `Tensor<T> bias_add(const Tensor<T>& other) const`
- 使用 `operator+` 实现广播加法

### softmax

- `Tensor<T> softmax(size_t dim) const`
- 对指定维度执行 softmax
- 返回同形状张量

### argmax

- `size_t argmax() const`
- 返回张量中最大元素的扁平索引

## 元素级运算

- `operator+(const Tensor<T>& other)`：支持广播加法
- `operator-(const Tensor<T>& other)`：元素级减法
- `operator*(T scalar)` / `operator/(T scalar)`：标量乘除
- `operator*(const Tensor<T>& other)`：要求形状一致的逐元素乘法
- `operator+(T scalar)` / `operator-(T scalar)`：标量加减

## 广播与扩展

- `broadcast(const vector<size_t>& new_shape) const`
- 自动扩展单例维度以匹配目标形状

## 参数初始化工具

- `he_normal_init(size_t in_feature, size_t out_feature)`：He 正态初始化，用于权重
- `zero_init(size_t in_feature, size_t out_feature)`：零值初始化，用于偏置

## 额外工具

- `index_to_coordinates(size_t idx) const`：将扁平索引转换为多维坐标

## 说明与使用提示

- `matmul` 要求输入张量的倒数第二维与另一个张量的倒数第二维匹配
- `softmax` 在计算时对每个“外部块”进行数值稳定的归一化
- `concat` 要求除拼接维度外，其余维度一致
- `broadcast` 要求每个维度要么相等，要么为 1

该张量库是 ViT 模型前向推理的基础，支持 `Model.h` 中的线性变换、attention 计算和 reshape/permute 操作。
