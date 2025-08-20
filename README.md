# VAMamba: An Efficient Visual Adaptive Mamba for Image Restoration


# 🚀 Overview framework
<img width="1196" height="693" alt="image" src="https://github.com/user-attachments/assets/a4c67afe-9957-4b6e-bd47-325c9f104018" />
Mamba-based image restoration methods
 have shown promising results. However, these methods face
 critical performance limitations due to fixed scanning patterns
 and inefficient feature utilization. Traditional Mamba architec
tures rely on predetermined scanning paths that cannot adapt
 to varying image degradation patterns. This rigidity severely
 constrains their restoration capabilities and computational ef
f
 iciency. To address these fundamental limitations, we propose
 VAMamba, a Visual Adaptive Mamba architecture with two key
 innovations. Our first contribution introduces QCLAM (Queue
based Cache Low-rank Adaptive Memory), a novel module
 that enhances feature learning efficiency. QCLAM employs a
 FIFO queue-based cache system that stores historical features.
 We compute similarity scores between current LoRA-adapted
 features and cached features to perform intelligent fusion. The
 fused features are then enqueued while the oldest features are
 dequeued, maintaining optimal memory usage and enabling
 dynamic feature reuse. Our second contribution presents GPS
SS2D (Greedy Path Scan SS2D), an adaptive scanning mechanism
 that replaces fixed scanning patterns. GPS-SS2D employs a
 Vision Transformer to generate score maps that evaluate pixel
 importance. A greedy strategy then plans optimal forward and
 backward scanning paths based on these scores. These learned
 paths serve as dynamic scanning trajectories for the SS2D
 mechanism, enabling targeted feature extraction. The integration
 of these innovations allows VAMamba to adaptively focus on
 degraded regions while maintaining computational efficiency.
 Extensive experiments demonstrate that VAMamba achieves su
perior performance across multiple image restoration tasks. Our
 method significantly outperforms existing approaches in both
 restoration quality and computational efficiency, establishing new
 benchmarks for adaptive image restoration.

# 🚀 Comparison of Scanning Methods
<img width="1196" height="274" alt="image" src="https://github.com/user-attachments/assets/3c7a0cbb-a170-4d7f-9d5e-118249df5d2c" />
Our
 GPS-SS2D method identifies regions with varying degradation
 severity using ViT-generated importance score maps and plans
 optimal scanning trajectories through greedy path selection.
 The numbered sequence (1→2→3) in the figure illustrates how
 our method prioritizes high-importance regions while main
taining spatial locality when possible. This adaptive approach
 offers three key advantages: (1) Content-aware processing:
 prioritizes regions based on their restoration importance for
 optimal resource allocation; (2) Dynamic path optimization:
 ensures sequential processing of neighboring high-importance
 regions; (3) Global-local balance: maintains global contextual
 awareness while focusing on locally important regions.

# 🔥 Installation

```bash
# create an environment with python >= 3.10
conda create -n vamamba python=3.10
conda activate vamamba
pip install -r requirements.txt
```

# 🔥 Train
## Step1 : Put training data into `datasets/train`
```bash
datasets
        |--train
               |--LQ
               |--HQ
```
## Step2 : Run code

```bash
torchrun --nproc_per_node=2 train.py -opt options/train/train_VAMamba.yml
```
