#include<vector>
#include<cmath>
#include<random>

template <class T>
class Tensor {
private:
    std::vector<T> _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
public:
    Tensor() = default;
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const T& init_val);
    size_t size() const;
    size_t rank() const;
    const std::vector<size_t>& shape() const;
    T& operator[](size_t idx);
    const T& operator[](size_t idx) const;
    template <class... Index>
    T& operator()(Index... idx);
    template <class... Index>
    const T& operator()(Index... idx) const;
    void reshape(const std::vector<size_t>& new_shape);
    Tensor<T> reshaped(const std::vector<size_t>& new_shape) const;
    Tensor<T> permute(const std::vector<size_t>& dims) const;
    Tensor<T> transpose() const;
    Tensor<T> unsqueeze(size_t dim) const;
    Tensor<T> squeeze(size_t dim) const;
    Tensor<T> slice(size_t dim, size_t begin, size_t end) const;
    static Tensor<T> concat(const std::vector<Tensor<T>>& tensors, size_t dim);
    Tensor<T> bias_add(const Tensor<T>& other) const;
    Tensor<T> matmul(const Tensor<T>& other) const;
    Tensor<T> softmax(size_t dim) const;
    size_t argmax() const;
    std::vector<size_t> Tensor<T>::index_to_coordinates(size_t idx) const;
    void he_normal_init(size_t in_feature,size_t out_feature)
    {
        *this = Tensor<T>({in_feature, out_feature});
        float std_dev = std::sqrt(2.0f / static_cast<float>(in_feature));

        std::random_device rd;
        std::mt19937 rnd(rd());
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        for(auto &i : _data)i = normal_dist(rnd) * std_dev;
    }
    void zero_init(size_t in_feature,size_t out_feature)
    {
        *this = Tensor<T>({in_feature, out_feature});
        for(auto &i : _data)i = 0;
    }
};