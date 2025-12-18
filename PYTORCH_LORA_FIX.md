# PyTorch LoRA 训练修复文档

## 问题总结

在 PyTorch 版本的 LoRA 训练中发现以下问题：

1. **JAX vs PyTorch LoRAConfig 混用**：`pi0_pytorch.py` 使用了 JAX 版本的 gemma config
2. **transformers 文件未替换**：修改的 transformers 文件没有复制到安装目录
3. **LoRALinear.base_layer 参数被错误解冻**：LoRA 的基础权重应该冻结但没有
4. **参数匹配不精确**：`"paligemma" in name` 会错误匹配 `gemma_expert` 的参数
5. **LoRALinear.weight 属性访问错误**：LoRALinear 没有直接的 `weight` 属性

## 完整修复方案

### 1. 创建 PyTorch 版本的 Gemma 配置

**新文件**：`src/openpi/models_pytorch/gemma_config.py`

```python
from openpi.models_pytorch.lora import LoRAConfig  # PyTorch 版本

def get_config(variant):
    if variant == "gemma_300m_lora":
        return Config({
            ...
            "lora_configs": {
                "attn": LoRAConfig(rank=32, alpha=32.0),
                "ffn": LoRAConfig(rank=32, alpha=32.0),
            },
        })
```

### 2. 修改模型创建代码

**文件**：`src/openpi/models_pytorch/pi0_pytorch.py`

```python
# 添加导入
import openpi.models_pytorch.gemma_config as _gemma_pytorch

# 修改配置获取
paligemma_config = _gemma_pytorch.get_config(config.paligemma_variant)
action_expert_config = _gemma_pytorch.get_config(config.action_expert_variant)
```

### 3. 添加权重访问辅助函数

**所有相关文件**中添加：

```python
def _get_layer_weight(layer):
    """Get weight from layer, handling both nn.Linear and LoRALinear."""
    if hasattr(layer, 'base_layer'):
        return layer.base_layer.weight
    return layer.weight
```

然后替换所有 `layer.weight` 访问为 `_get_layer_weight(layer)`。

### 4. 修复参数冻结逻辑

**文件**：`scripts/train_pytorch.py`

```python
for name, param in model_to_freeze.named_parameters():
    # 1. 解冻 LoRA 参数
    if "lora_A" in name or "lora_B" in name:
        param.requires_grad = True
        continue
    
    # 2. 跳过 base_layer（关键！）
    if "base_layer" in name:
        continue
    
    # 3. 解冻投影层
    if "action_in_proj" in name or "action_out_proj" in name:
        param.requires_grad = True
        continue
    
    # 4. 精确匹配 paligemma（避免匹配 gemma_expert）
    if (not freeze_paligemma) and ("paligemma.language_model" in name or (".language_model" in name and "gemma_expert" not in name)):
        param.requires_grad = True
        continue
    
    # 5. 解冻 action_expert（如果不使用 LoRA）
    if (not freeze_action_expert) and "gemma_expert" in name:
        param.requires_grad = True
        continue
```

### 5. 更新 transformers 安装

**每次修改 transformers_replace 后运行**：

```bash
./update_transformers.sh
```

或手动：

```bash
cp -r src/openpi/models_pytorch/transformers_replace/models/gemma/* .venv/lib/python3.11/site-packages/transformers/models/gemma/
```

## 修复后的效果

### 配置 1：只对 action_expert 使用 LoRA

```python
paligemma_variant="gemma_2b"           # 全量微调
action_expert_variant="gemma_300m_lora"  # LoRA
```

**可训练参数：~3.2B (~91%)**
- Paligemma: 3.2B（全量）
- Action Expert LoRA: 13.86M
- 其他：3.25M

### 配置 2：两者都使用 LoRA（推荐！）

```python
paligemma_variant="gemma_2b_lora"        # LoRA
action_expert_variant="gemma_300m_lora"  # LoRA
```

**可训练参数：~37M (~1%)**
- Paligemma LoRA: 19.61M
- Action Expert LoRA: 13.86M
- 其他：3.25M

## 验证步骤

1. 运行训练：

```bash
uv run scripts/train_pytorch.py pi0_libero_lora_pytorch --exp_name test --save_interval 1000
```

2. 检查日志应显示：

```
✅ Found 504 LoRA parameters after model creation
✅ Freeze paligemma: True, freeze action expert: True  
✅ LoRA fine-tuning: 36,720,672 trainable / 3,534,844,688 total (1.04%)
```

3. 推理测试应该不会出现 AttributeError

## 常见问题

### Q: 为什么需要复制 transformers 文件？

A: 我们对 transformers 的 Gemma 模型做了修改（添加 LoRA 支持），需要替换安装的版本。

### Q: 每次修改代码都需要运行 update_transformers.sh 吗？

A: 只有修改 `src/openpi/models_pytorch/transformers_replace/` 中的文件时才需要。

### Q: 如何验证 transformers 是否正确更新？

A: 运行：

```bash
grep -q "_get_layer_weight" .venv/lib/python3.11/site-packages/transformers/models/gemma/modeling_gemma.py && echo "✅ Updated" || echo "❌ Not updated"
```

## 相关文件

- `src/openpi/models_pytorch/gemma_config.py` - PyTorch Gemma 配置
- `src/openpi/models_pytorch/lora.py` - PyTorch LoRA 实现
- `src/openpi/models_pytorch/pi0_pytorch.py` - PI0 PyTorch 模型
- `src/openpi/models_pytorch/gemma_pytorch.py` - Gemma 包装器
- `src/openpi/models_pytorch/transformers_replace/models/gemma/` - 修改的 transformers 文件
- `scripts/train_pytorch.py` - 训练脚本
- `update_transformers.sh` - 自动更新脚本







