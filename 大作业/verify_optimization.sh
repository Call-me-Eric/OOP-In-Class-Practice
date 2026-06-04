#!/bin/bash
# verify_optimization.sh - 验证优化是否正确应用

echo "======================================================"
echo "   MNIST ViT 优化验证脚本"
echo "======================================================"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")" || exit 1

# 检查 main.cpp 中的集合学习是否启用
echo -e "\n${YELLOW}[检查 1]${NC} 集合学习是否启用..."
if grep -q "img_center = raw" main.cpp && grep -q "logits_ctr = vit.forward" main.cpp; then
    echo -e "${GREEN}✓${NC} 集合学习已启用"
else
    echo -e "${RED}✗${NC} 集合学习未找到"
fi

# 检查置信度投票机制
echo -e "\n${YELLOW}[检查 2]${NC} 置信度投票机制..."
if grep -q "max_ctr > max_std" main.cpp; then
    echo -e "${GREEN}✓${NC} 置信度投票机制已实现"
else
    echo -e "${RED}✗${NC} 置信度投票未找到"
fi

# 检查 evaluate_test_data.py 中的详细统计
echo -e "\n${YELLOW}[检查 3]${NC} 评估脚本改进..."
if grep -q "errors_by_digit" evaluate_test_data.py; then
    echo -e "${GREEN}✓${NC} 按数字统计已实现"
else
    echo -e "${RED}✗${NC} 按数字统计未找到"
fi

# 检查新增文件
echo -e "\n${YELLOW}[检查 4]${NC} 新增文件..."

files=(
    "OPTIMIZATION_GUIDE.md"
    "QUICK_START.md"
    "advanced_optimizer.py"
    "compare_results.py"
    "IMPROVEMENTS_SUMMARY.md"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (缺失)"
        all_exist=false
    fi
done

# 检查编译可行性
echo -e "\n${YELLOW}[检查 5]${NC} 编译检查..."
if command -v g++ &> /dev/null; then
    echo -e "${GREEN}✓${NC} g++ 已安装"
    
    # 尝试编译
    echo "正在编译 main.cpp..."
    if g++ main.cpp -o main_test -O2 -std=c++20 2>/dev/null; then
        echo -e "${GREEN}✓${NC} 编译成功"
        rm -f main_test
    else
        echo -e "${YELLOW}⚠${NC} 编译有警告或错误（可能需要 C++17）"
        if g++ main.cpp -o main_test -O2 -std=c++17 2>/dev/null; then
            echo -e "${GREEN}✓${NC} 使用 C++17 编译成功"
            rm -f main_test
        fi
    fi
else
    echo -e "${RED}✗${NC} g++ 未安装"
fi

# 检查 Python 环境
echo -e "\n${YELLOW}[检查 6]${NC} Python 环境..."
if command -v python3 &> /dev/null; then
    py_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python $py_version"
else
    echo -e "${RED}✗${NC} Python 未安装"
fi

# 检查测试数据
echo -e "\n${YELLOW}[检查 7]${NC} 测试数据..."
if [ -d "mnist_raw/test" ]; then
    sample_count=$(ls -1 mnist_raw/test/*.raw 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} 找到 $sample_count 个测试样本"
else
    echo -e "${RED}✗${NC} 测试数据目录不存在"
fi

# 检查权重文件
echo -e "\n${YELLOW}[检查 8]${NC} 权重文件..."
if [ -d "weights_finetuned_norm" ]; then
    weight_count=$(ls -1 weights_finetuned_norm/*.wts 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} 找到 $weight_count 个权重文件"
else
    echo -e "${RED}✗${NC} 权重目录不存在"
fi

echo -e "\n======================================================"
echo "验证完成！"
echo -e "======================================================"
echo -e "\n${YELLOW}下一步：${NC}"
echo "1. 编译: g++ main.cpp -o main -O2 -std=c++20"
echo "2. 测试: ./main mnist_raw/test/test_0000_label_1.raw weights_finetuned_norm 1"
echo "3. 评估: python evaluate_test_data.py"
echo ""
