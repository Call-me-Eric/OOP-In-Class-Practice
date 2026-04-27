#include "Tensor.h"
#include<vector>
#include<iostream>
#include<random>
using std::cerr;

class Layer//layer抽象基类
{
public:
    virtual Tensor<float> forward(const Tensor<float>& x) = 0;
    virtual ~Layer() = default;
};

class Linear : public Layer//linear线性层类
{
private:
    Tensor<float> _weight;
    Tensor<float> _bias;
    size_t in_features, out_features;
public:
    //做了两种接口
    Linear(size_t in_f = 0,size_t out_f = 0):in_features(in_f),out_features(out_f)//一种是直接传参数然后构造一套默认的仿射变换
    {
        _weight.he_normal_init(in_f,out_f);
        _bias.zero_init(in_f,out_f);
    }
    Linear(Tensor<float> _weight, Tensor<float> _bias):_weight(_weight),_bias(_bias){}//另一种是指定权重和偏移
    Tensor<float> forward(const Tensor<float>& x)
    {
        Tensor<float> output = x.matmul(_weight);
        return output.bias_add(_bias);
    }
};

/*
PatchEmbedding类通过将1\times 28\times 28的图像转换成16个7\times 7的Patch，映射到hidden_dim=32维向量上，以供下一步Transformer学习。
同时需要加一个CLS token用于学习。
输入[B, 28, 28],输出[B, patch_num=16 + CLS_num=1, hidden_dim=32]
*/
class PatchEmbedding : public Layer
{
private:
    static const int PATCH_SIZE = 7;
    static const int HIDDEN_DIM = 32;
    static const int NUM_PATCHES = 16;
    Linear projection;
    Tensor<float> cls_token;
    Tensor<float> pos_embedding;
public:
    PatchEmbedding() : projection(PATCH_SIZE * PATCH_SIZE, HIDDEN_DIM)
    {
        cls_token = Tensor<float>({1, 1, HIDDEN_DIM});
        pos_embedding = Tensor<float>({1,NUM_PATCHES + 1,HIDDEN_DIM});
    }
    Tensor<float> forward(const Tensor<float>& x)
    {
        size_t B = x.shape()[0];
        // 将输入 [B, 28, 28] 重塑为 [B, 16, 49]

        Tensor<float> patches = x.reshaped({B, NUM_PATCHES, PATCH_SIZE * PATCH_SIZE});
        // 展平为 [B*16, 49] 以通过线性层
        patches = patches.reshaped({B * NUM_PATCHES, PATCH_SIZE * PATCH_SIZE});
        // 通过投影层得到 [B*16, 32]
        Tensor<float> embedded = projection.forward(patches);
        // 重塑回 [B, 16, 32]
        embedded = embedded.reshaped({B, NUM_PATCHES, HIDDEN_DIM});
        // 手动广播 CLS token 到 [B, 1, 32]
        Tensor<float> cls_expanded({B, 1, HIDDEN_DIM});
        for (size_t b = 0; b < B; ++b) {
            for (size_t j = 0; j < HIDDEN_DIM; ++j) {
                cls_expanded(b, 0, j) = cls_token(0, 0, j);
            }
        }
        // 使用 concat 在维度 1 上拼接，得到 [B, 17, 32]
        Tensor<float> with_cls = Tensor<float>::concat({embedded, cls_expanded}, 1);
        // 添加位置嵌入
        Tensor<float> output = with_cls + pos_embedding;
        return output;
    }
};
