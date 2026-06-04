#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_local_mnist.py

使用 mnist_raw/test 中的测试样本 + MNIST 原始数据来针对性微调 ViT 权重。
策略：80%测试数据用于训练（含数据增强），20%用于验证，配合 MNIST 原始数据防止过拟合。

用法示例：
    # 仅用测试数据微调（快速适应测试分布）
    python3 train_local_mnist.py --weights-dir weights_finetuned_norm \
        --out-dir weights_finetuned_norm --epochs 15 --lr 5e-4

    # 结合 MNIST 原始数据 + 测试数据
    python3 train_local_mnist.py --weights-dir weights_finetuned_norm \
        --out-dir weights_finetuned_norm --epochs 20 --lr 3e-4 --use-mnist 3000
"""

import argparse
from pathlib import Path
import numpy as np
import random


def load_raw_test_data(test_dir: Path):
    """从 mnist_raw/test 加载 raw 图像和标签"""
    labels_file = test_dir / 'labels.txt'
    if not labels_file.exists():
        raise FileNotFoundError(f'找不到标签文件: {labels_file}')

    images = []
    labels = []

    with labels_file.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            filename, label = parts
            filepath = test_dir / filename
            if not filepath.exists():
                continue

            # 读取 784 字节 raw 数据
            raw_bytes = filepath.read_bytes()
            if len(raw_bytes) != 784:
                print(f'[WARN] 跳过 {filename}: 大小={len(raw_bytes)}')
                continue

            img = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
            images.append(img)
            labels.append(int(label))

    images = np.array(images)  # [N, 784]
    labels = np.array(labels, dtype=np.int64)
    print(f'从 {test_dir} 加载了 {len(images)} 个测试样本')
    return images, labels


def add_data_augmentation(images: np.ndarray, labels: np.ndarray,
                          augment_factor: int = 2):
    """对训练数据做离线增强：轻微平移 + 缩放变体"""
    aug_images = [images]
    aug_labels = [labels]

    for _ in range(augment_factor):
        shifted = np.zeros_like(images)
        for i in range(len(images)):
            img_28 = images[i].reshape(28, 28)

            # 随机平移 -2~+2 像素
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)

            shifted_28 = np.zeros((28, 28), dtype=np.float32)
            x_start = max(0, dx)
            x_end = min(28, 28 + dx)
            y_start = max(0, dy)
            y_end = min(28, 28 + dy)

            src_x_start = max(0, -dx)
            src_x_end = min(28, 28 - dx)
            src_y_start = max(0, -dy)
            src_y_end = min(28, 28 - dy)

            shifted_28[y_start:y_end, x_start:x_end] = \
                img_28[src_y_start:src_y_end, src_x_start:src_x_end]

            # 轻微随机缩放 0.95~1.05
            scale = 1.0 + random.uniform(-0.05, 0.05)
            shifted[i] = shifted_28.reshape(784) * scale

        shifted = np.clip(shifted, 0, 255)
        aug_images.append(shifted)
        aug_labels.append(labels)

    return np.concatenate(aug_images), np.concatenate(aug_labels)


def compute_class_weights(labels: np.ndarray, num_classes: int = 10):
    """计算类别权重，给样本少的类别更高权重"""
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes
    return weights


def main():
    parser = argparse.ArgumentParser(description='针对性微调 ViT 权重')
    parser.add_argument('--weights-dir', default='weights_finetuned_norm')
    parser.add_argument('--out-dir', default='weights_finetuned_norm')
    parser.add_argument('--test-dir', default='mnist_raw/test')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='验证集比例 (0=全部训练)')
    parser.add_argument('--use-mnist', type=int, default=0,
                        help='混入 MNIST 原始训练样本数量 (0=不用)')
    parser.add_argument('--augment', type=int, default=3,
                        help='数据增强倍数 (0=不增强)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        print('需要安装 torch 才能训练: pip install torch')
        raise

    import fine_tune_weights as fw_mod
    import importlib.util, types

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)
    test_dir = Path(args.test_dir)

    # ============================================================
    #  1. 加载测试数据
    # ============================================================
    print('=' * 60)
    print('  加载 mnist_raw/test 数据')
    print('=' * 60)
    test_images, test_labels = load_raw_test_data(test_dir)

    # 打印类别分布
    for d in range(10):
        cnt = (test_labels == d).sum()
        print(f'  数字 {d}: {cnt} 个样本')

    # ============================================================
    #  2. 数据预处理: 归一化 + 标准化
    # ============================================================
    # raw 值是 0-255，先归一化到 [0,1]
    test_images = test_images / 255.0

    # ============================================================
    #  3. 划分训练/验证集
    # ============================================================
    n_total = len(test_images)
    indices = np.random.permutation(n_total)

    if args.val_split > 0:
        n_val = int(n_total * args.val_split)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        print(f'\n训练集: {len(train_idx)} 样本, 验证集: {len(val_idx)} 样本')
    else:
        train_idx = indices
        val_idx = np.array([], dtype=int)
        print(f'\n训练集: {len(train_idx)} 样本 (全部用于训练)')

    X_train_raw = test_images[train_idx]
    y_train = test_labels[train_idx]

    # ============================================================
    #  4. 数据增强
    # ============================================================
    if args.augment > 0:
        print(f'数据增强: ×{args.augment + 1} (原始 + {args.augment} 种变体)')
        X_train_raw, y_train = add_data_augmentation(
            X_train_raw * 255.0, y_train, args.augment)
        X_train_raw = X_train_raw / 255.0

    # ============================================================
    #  5. 混入 MNIST 原始数据（防止过拟合）
    # ============================================================
    if args.use_mnist > 0:
        import mnist_optimize as mn_mod
        print(f'混入 {args.use_mnist} 个 MNIST 原始训练样本...')
        mnist_imgs, mnist_labels, _, _ = mn_mod.download_mnist(Path('mnist_data'))
        mnist_imgs = mnist_imgs[:args.use_mnist].astype(np.float32) / 255.0
        mnist_labels = mnist_labels[:args.use_mnist].astype(np.int64)
        # 去掉多余的通道维度: 统一为 (N, 28, 28)
        if mnist_imgs.ndim == 4:
            mnist_imgs = mnist_imgs.squeeze(1)  # (N,1,28,28) → (N,28,28)
        elif mnist_imgs.ndim == 2:
            mnist_imgs = mnist_imgs.reshape(-1, 28, 28)  # (N,784) → (N,28,28)

        # test data: (N, 784) → (N, 28, 28) for concat
        X_train_reshaped = X_train_raw.reshape(-1, 28, 28)

        X_train_raw = np.concatenate([X_train_reshaped, mnist_imgs])
        y_train = np.concatenate([y_train, mnist_labels])
        print(f'合并后训练集: {len(X_train_raw)} 样本')

    # ============================================================
    #  6. 类别权重（处理不平衡）
    # ============================================================
    class_weights = compute_class_weights(y_train)
    print(f'类别权重: {[f"{w:.2f}" for w in class_weights]}')

    # ============================================================
    #  7. 构建 PyTorch 模型
    # ============================================================
    print('\n' + '=' * 60)
    print('  构建模型并加载权重')
    print('=' * 60)

    weights = fw_mod.load_weights_dict(weights_dir)

    spec = importlib.util.spec_from_loader('vt_model', loader=None)
    vt_mod = types.ModuleType('vt_model')
    exec(fw_mod.MODEL_CODE, vt_mod.__dict__)
    Vit = vt_mod.VisionTransformer

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = Vit().to(device)

    def to_t(t):
        return torch.from_numpy(t.copy()).to(torch.float32)

    # 注入权重
    model.patch_proj.weight.data.copy_(to_t(weights['patch.weight']).T)
    model.patch_proj.bias.data.copy_(to_t(weights['patch.bias']).reshape(-1))
    model.cls_token.data.copy_(to_t(weights['cls_token'].reshape(1, 1, 32)))
    model.pos_embed.data.copy_(to_t(weights['pos_embed']))

    for i in range(2):
        blk = model.blocks[i]
        p = f'blocks.{i}'
        blk.q.weight.data.copy_(to_t(weights[f'{p}.attn.q.weight']).T)
        blk.q.bias.data.copy_(to_t(weights[f'{p}.attn.q.bias']).reshape(-1))
        blk.k.weight.data.copy_(to_t(weights[f'{p}.attn.k.weight']).T)
        blk.k.bias.data.copy_(to_t(weights[f'{p}.attn.k.bias']).reshape(-1))
        blk.v.weight.data.copy_(to_t(weights[f'{p}.attn.v.weight']).T)
        blk.v.bias.data.copy_(to_t(weights[f'{p}.attn.v.bias']).reshape(-1))
        blk.out.weight.data.copy_(to_t(weights[f'{p}.attn.o.weight']).T)
        blk.out.bias.data.copy_(to_t(weights[f'{p}.attn.o.bias']).reshape(-1))
        blk.mlp.fc1.weight.data.copy_(to_t(weights[f'{p}.mlp.fc1.weight']).T)
        blk.mlp.fc1.bias.data.copy_(to_t(weights[f'{p}.mlp.fc1.bias']).reshape(-1))
        blk.mlp.fc2.weight.data.copy_(to_t(weights[f'{p}.mlp.fc2.weight']).T)
        blk.mlp.fc2.bias.data.copy_(to_t(weights[f'{p}.mlp.fc2.bias']).reshape(-1))
        try:
            blk.norm1.weight.data.copy_(to_t(weights[f'{p}.norm1.gamma']).reshape(-1))
            blk.norm1.bias.data.copy_(to_t(weights[f'{p}.norm1.beta']).reshape(-1))
            blk.norm2.weight.data.copy_(to_t(weights[f'{p}.norm2.gamma']).reshape(-1))
            blk.norm2.bias.data.copy_(to_t(weights[f'{p}.norm2.beta']).reshape(-1))
        except Exception:
            pass

    model.head.weight.data.copy_(to_t(weights['head.weight']).T)
    model.head.bias.data.copy_(to_t(weights['head.bias']).reshape(-1))

    # ============================================================
    #  8. 准备 DataLoader
    # ============================================================
    # ============================================================
    #  8. 准备 DataLoader（模型期望 (B, 28, 28)）
    # ============================================================
    # 训练数据: 标准化 + reshape → (N, 28, 28)
    X_train = (X_train_raw - 0.1307) / 0.3081
    X_train = X_train.reshape(-1, 28, 28).astype(np.float32)
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # 验证集: 同样标准化 + reshape
    if len(val_idx) > 0:
        X_val_raw = test_images[val_idx]
        y_val = test_labels[val_idx]
        X_val = (X_val_raw - 0.1307) / 0.3081
        X_val = X_val.reshape(-1, 28, 28).astype(np.float32)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val.astype(np.int64))
        val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None

    # ============================================================
    #  9. 训练配置
    # ============================================================
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    print('\n' + '=' * 60)
    print(f'  开始训练 ({args.epochs} epochs, lr={args.lr}, bs={args.batch_size})')
    print('=' * 60)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = correct / total * 100
        avg_loss = running_loss / len(train_loader)

        # 验证
        val_str = ''
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            val_acc = val_correct / val_total * 100
            val_str = f' val_acc={val_acc:.2f}%'

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'epoch {epoch+1:2d}/{args.epochs} | loss={avg_loss:.4f} | '
              f'train_acc={train_acc:.2f}%{val_str} | lr={scheduler.get_last_lr()[0]:.2e}')

    # 恢复最佳模型
    if best_state is not None:
        print(f'\n恢复最佳验证模型 (val_acc={best_val_acc:.2f}%)')
        model.load_state_dict(best_state)

    # ============================================================
    #  10. 导出权重
    # ============================================================
    print('\n' + '=' * 60)
    print('  导出优化后的权重')
    print('=' * 60)

    new_w = {}
    new_w['patch.weight'] = model.patch_proj.weight.data.cpu().numpy().T
    new_w['patch.bias'] = model.patch_proj.bias.data.cpu().numpy().reshape(1, -1)
    new_w['cls_token'] = model.cls_token.data.cpu().numpy().reshape(1, 1, -1)
    new_w['pos_embed'] = model.pos_embed.data.cpu().numpy().reshape(1, 17, -1)

    for i in range(2):
        blk = model.blocks[i]
        p = f'blocks.{i}'
        new_w[f'{p}.attn.q.weight'] = blk.q.weight.data.cpu().numpy().T
        new_w[f'{p}.attn.q.bias'] = blk.q.bias.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.attn.k.weight'] = blk.k.weight.data.cpu().numpy().T
        new_w[f'{p}.attn.k.bias'] = blk.k.bias.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.attn.v.weight'] = blk.v.weight.data.cpu().numpy().T
        new_w[f'{p}.attn.v.bias'] = blk.v.bias.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.attn.o.weight'] = blk.out.weight.data.cpu().numpy().T
        new_w[f'{p}.attn.o.bias'] = blk.out.bias.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.mlp.fc1.weight'] = blk.mlp.fc1.weight.data.cpu().numpy().T
        new_w[f'{p}.mlp.fc1.bias'] = blk.mlp.fc1.bias.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.mlp.fc2.weight'] = blk.mlp.fc2.weight.data.cpu().numpy().T
        new_w[f'{p}.mlp.fc2.bias'] = blk.mlp.fc2.bias.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.norm1.gamma'] = blk.norm1.weight.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.norm1.beta'] = blk.norm1.bias.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.norm2.gamma'] = blk.norm2.weight.data.cpu().numpy().reshape(1, -1)
        new_w[f'{p}.norm2.beta'] = blk.norm2.bias.data.cpu().numpy().reshape(1, -1)

    new_w['head.weight'] = model.head.weight.data.cpu().numpy().T
    new_w['head.bias'] = model.head.bias.data.cpu().numpy().reshape(1, -1)

    # 备份旧权重
    backup_dir = out_dir.parent / (out_dir.name + '_backup')
    if out_dir.exists() and not backup_dir.exists():
        import shutil
        shutil.copytree(out_dir, backup_dir)
        print(f'旧权重已备份到: {backup_dir}')

    fw_mod.save_weights_dict(out_dir, new_w)
    print(f'优化后的权重已保存到: {out_dir}')
    print(f'训练集准确率: {train_acc:.2f}%')
    if val_loader:
        print(f'验证集准确率: {best_val_acc:.2f}%')


if __name__ == '__main__':
    main()
