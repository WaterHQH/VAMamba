# MambaIR LoRA和Cache功能使用说明

## 概述

本文档介绍如何在MambaIR模型中添加LoRA（Low-Rank Adaptation）和Cache（缓存）机制。这些功能可以帮助：

1. **LoRA**: 通过低秩适应降低计算复杂度，提高训练效率
2. **Cache**: 缓存历史特征，减少重复计算，提升推理速度

## 功能特性

### LoRA (Low-Rank Adaptation)
- **降维处理**: 通过低秩矩阵分解降低特征维度
- **可学习参数**: 包含A和B两个低秩矩阵，可训练
- **权重衰减**: 支持dropout防止过拟合
- **灵活配置**: 可调整rank大小和alpha参数

### Cache (特征缓存)
- **历史特征存储**: 缓存历史处理的特征
- **相似度匹配**: 使用余弦相似度找到相似特征
- **自适应权重**: 学习缓存特征的融合权重
- **内存管理**: 支持最大缓存大小限制

## 架构修改

### SS2D模块修改
在SS2D的forward方法中，在SiLU激活函数后添加了LoRA和Cache处理：

```python
def forward(self, x: torch.Tensor, cache_key=None, **kwargs):
    # ... 原有代码 ...
    x = self.act(self.conv2d(x))
    
    # 新增：LoRA和Cache处理
    if self.use_lora_cache and self.lora_cache is not None:
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()
        x_processed = self.lora_cache(x_reshaped, cache_key)
        x = x_processed.permute(0, 3, 1, 2).contiguous()
    
    # ... 后续处理 ...
```

### 双分支结构
SS2D现在包含两个分支：
1. **分支1 (z分支)**: 门控信号分支
2. **分支2 (x分支)**: 主要处理分支，包含：
   - Linear投影
   - Conv2D卷积
   - SiLU激活
   - **LoRA降维** ← 新增
   - **Cache缓存** ← 新增
   - 2D-SSM状态空间模型

## 配置参数

### 网络配置
```yaml
network_g:
  type: MambaIR
  # 基础参数
  upscale: 1
  in_chans: 3
  img_size: 64
  embed_dim: 180
  depths: [4, 4, 4, 4, 4, 4]
  mlp_ratio: 1.2
  d_state: 16
  
  # LoRA和Cache参数
  use_lora_cache: true      # 是否启用LoRA和Cache
  lora_rank: 16            # LoRA的rank大小
  cache_size: 5            # 缓存大小
```

### 高级配置
```python
# 在SS2D初始化时可以设置更多参数
SS2D(
    d_model=180,
    use_lora_cache=True,
    lora_rank=16,          # LoRA rank
    cache_size=5,          # 缓存大小
    # 其他参数...
)
```

## 使用方法

### 1. 基本使用
```python
from basicsr.archs import build_network

# 构建模型
model_config = {
    'type': 'MambaIR',
    'use_lora_cache': True,
    'lora_rank': 16,
    'cache_size': 5,
    # 其他参数...
}

model = build_network(model_config)

# 前向传播
x = torch.randn(1, 3, 256, 256)
output = model(x, cache_key="unique_key")
```

### 2. 训练配置
使用提供的配置文件：
```bash
python basicsr/train.py -opt options/train/train_MambaIR_LoRA_Cache.yml
```

### 3. 推理使用
```python
# 加载训练好的模型
model = build_network(model_config)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['params_ema'])

# 推理
with torch.no_grad():
    output = model(input_image, cache_key="inference_1")
```

## 性能优化

### 1. 内存优化
- **缓存清理**: 定期清理缓存以释放内存
- **相似度阈值**: 调整相似度阈值以平衡精度和速度
- **缓存大小**: 根据可用内存调整缓存大小

### 2. 计算优化
- **LoRA rank**: 较小的rank可以减少计算量
- **缓存命中率**: 提高缓存命中率可以减少重复计算
- **批处理**: 支持批处理以提高GPU利用率

### 3. 精度优化
- **特征融合**: 自适应权重融合缓存特征
- **残差连接**: 保持原始特征信息
- **梯度裁剪**: 防止梯度爆炸

## 监控和调试

### 1. 缓存统计
```python
# 获取缓存统计信息
for name, module in model.named_modules():
    if hasattr(module, 'cache') and module.cache is not None:
        print(f"{name}: 缓存大小 = {len(module.cache)}")
```

### 2. LoRA参数统计
```python
# 统计LoRA参数
lora_params = 0
for name, module in model.named_modules():
    if 'lora' in name.lower():
        lora_params += sum(p.numel() for p in module.parameters())
print(f"LoRA参数数量: {lora_params:,}")
```

### 3. 性能监控
```python
import time

# 测试推理时间
start_time = time.time()
output = model(input_tensor, cache_key="test")
end_time = time.time()
print(f"推理时间: {(end_time - start_time)*1000:.2f} ms")
```

## 注意事项

### 1. 兼容性
- 保持与原始MambaIR的兼容性
- 可以通过`use_lora_cache=False`禁用新功能
- 支持加载原始权重文件

### 2. 内存使用
- LoRA会增加少量参数
- Cache会占用额外内存
- 建议根据GPU内存调整配置

### 3. 训练稳定性
- 使用AdamW优化器
- 适当的权重衰减
- 梯度裁剪防止梯度爆炸

## 故障排除

### 1. 内存不足
- 减少`cache_size`
- 降低`lora_rank`
- 使用梯度检查点

### 2. 训练不稳定
- 调整学习率
- 增加权重衰减
- 检查梯度范数

### 3. 精度下降
- 增加`lora_rank`
- 调整相似度阈值
- 检查特征融合权重

## 未来改进

1. **动态rank**: 根据输入自适应调整LoRA rank
2. **多尺度缓存**: 支持不同尺度的特征缓存
3. **注意力缓存**: 在注意力机制中应用缓存
4. **分布式缓存**: 支持多GPU间的缓存共享 