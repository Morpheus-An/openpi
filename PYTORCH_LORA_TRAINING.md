# PyTorch LoRA 微调使用指南

本指南说明如何使用 PyTorch 版本的 LoRA 在 Libero 数据集上微调 pi0_base 模型。

## 前置条件

1. **确保已转换 PyTorch checkpoint**
   
   如果还没有 pi0_base 的 PyTorch checkpoint，需要先从 JAX checkpoint 转换：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name pi0_base \
       --checkpoint_dir /path/to/jax/pi0_base/checkpoint \
       --output_path /mars_data_2/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch
   ```

2. **检查 checkpoint 路径**
   
   确保配置中的 `pytorch_weight_path` 指向正确的 PyTorch checkpoint 目录。默认路径是：
   ```
   /mars_data_2/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch
   ```
   
   如果您的 checkpoint 在其他位置，可以：
   - 修改 `src/openpi/training/config.py` 中的 `pi0_libero_lora_pytorch` 配置
   - 或者在命令行中使用参数覆盖（见下方）

## 使用方法

### 方法 1: 使用预定义配置（推荐）

已添加了 `pi0_libero_lora_pytorch` 配置，可以直接使用：

```bash
# 单 GPU 训练
uv run scripts/train_pytorch.py pi0_libero_lora_pytorch --exp_name pi0_libero_lora_test

# 多 GPU 训练（单节点，例如 2 个 GPU）
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_libero_lora_pytorch --exp_name pi0_libero_lora_test

# 恢复训练
uv run scripts/train_pytorch.py pi0_libero_lora_pytorch --exp_name pi0_libero_lora_test --resume
```

### 方法 2: 使用命令行参数覆盖

如果 checkpoint 路径不同，可以在命令行中覆盖：

```bash
uv run scripts/train_pytorch.py pi0_libero_lora_pytorch \
    --exp_name pi0_libero_lora_test \
    --pytorch_weight_path /your/custom/path/to/pi0_base_pytorch
```

### 方法 3: 自定义配置

您也可以基于现有配置创建自己的配置，在 `src/openpi/training/config.py` 中添加：

```python
TrainConfig(
    name="my_pi0_libero_lora",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora", 
        action_expert_variant="gemma_300m_lora"
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    pytorch_weight_path="/your/path/to/pi0_base_pytorch",
    num_train_steps=30_000,
    ema_decay=None,
),
```

## 配置说明

### LoRA 配置

- `paligemma_variant="gemma_2b_lora"`: 对 PaliGemma 使用 LoRA，rank=16, alpha=16.0
- `action_expert_variant="gemma_300m_lora"`: 对 Action Expert 使用 LoRA，rank=32, alpha=32.0

### 自动参数冻结

训练脚本会自动检测 LoRA 配置并：
1. 冻结所有非 LoRA 参数（`requires_grad=False`）
2. 只训练 LoRA 参数（`lora_A` 和 `lora_B`）
3. 始终保留 action projection 层可训练（`action_in_proj`, `action_out_proj`, `state_proj`, `time_mlp` 等）

训练开始时会打印可训练参数的数量和百分比，例如：
```
LoRA fine-tuning: 15,234,432 trainable parameters out of 2,456,789,012 total (0.62%)
```

## 验证 LoRA 是否正常工作

训练开始后，检查日志中是否有以下信息：

1. **LoRA 检测成功**:
   ```
   LoRA fine-tuning detected. Freezing non-LoRA parameters...
   LoRA fine-tuning: X trainable parameters out of Y total (Z%)
   ```

2. **模型加载成功**:
   ```
   Loading weights from: /path/to/pi0_base_pytorch
   Loaded PyTorch weights from /path/to/pi0_base_pytorch
   ```

3. **训练正常进行**:
   - Loss 应该逐渐下降
   - 只有 LoRA 参数会更新（可以通过检查梯度确认）

## 常见问题

### 1. Checkpoint 路径不存在

如果遇到 `FileNotFoundError`，请检查：
- checkpoint 目录是否存在
- `model.safetensors` 文件是否在 checkpoint 目录中

### 2. LoRA 参数没有被冻结

如果所有参数都可训练，检查：
- 模型配置中的 variant 名称是否包含 "lora"
- 训练日志中是否显示 "LoRA fine-tuning detected"

### 3. 内存不足

LoRA 微调应该比全量微调使用更少内存。如果仍然遇到 OOM：
- 减小 `batch_size`
- 启用 gradient checkpointing（如果尚未启用）
- 使用更少的 GPU

## 监控训练

训练过程会记录到 wandb（如果配置了），可以监控：
- Loss 曲线
- 学习率
- 梯度范数
- 可训练参数数量

## 保存和加载

训练过程中的 checkpoint 会保存在：
```
./checkpoints/<exp_name>/
```

每个 checkpoint 包含：
- `model.safetensors`: 模型权重（包括 LoRA 参数）
- `optimizer.pt`: 优化器状态

恢复训练时使用 `--resume` 参数即可。

