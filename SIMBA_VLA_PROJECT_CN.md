# Simba + Completion-Augmented 3D Diffusion VLA

## 一句话定位

**在遮挡和稀疏 3D 观测下，用 Simba 几何补全恢复场景状态，再用 DP3 风格的 diffusion policy 做语言条件动作生成，验证点云补全对具身决策鲁棒性的系统性贡献。**

---

## 目录

1. [项目价值与投递方向](#1-项目价值与投递方向)
2. [系统架构](#2-系统架构)
3. [核心模块详解](#3-核心模块详解)
4. [训练与评估流程](#4-训练与评估流程)
5. [配置系统与多场景](#5-配置系统与多场景)
6. [实验设计与 Ablation](#6-实验设计与-ablation)
7. [简历写法建议](#7-简历写法建议)
8. [面试深度 Q&A（50+ 问题）](#8-面试深度-qa)
9. [快速上手](#9-快速上手)
10. [与前沿工作的对比定位](#10-与前沿工作的对比定位)

---

## 1. 项目价值与投递方向

### 为什么这个项目对简历有价值

这个项目天然卡在三个高薪方向的交叉点上：

| 投递方向 | 项目对应的核心能力 |
|---------|------------------|
| **具身智能 / 机器人** | 遮挡场景下语言条件操作策略；点云补全提升抓取/导航鲁棒性 |
| **世界模型** | "先恢复状态，再做规划" = 隐式世界模型的核心范式 |
| **自动驾驶** | 稀疏 LiDAR 补全 + command-conditioned waypoint prediction |

### 推荐简历项目标题

- **通用**: Completion-Augmented 3D Diffusion Policy for Occlusion-Robust VLA
- **偏具身**: Occlusion-Robust Robotic Manipulation via Point Cloud Completion + Diffusion Policy
- **偏自动驾驶**: Completion-Augmented LiDAR VLA for Command-Conditioned 3D Planning
- **偏世界模型**: Geometric State Recovery as Implicit World Model for 3D Vision-Language-Action

---

## 2. 系统架构

### 整体流程

```
┌──────────────────────────────────────────────────────────────────┐
│               Completion-Augmented 3D Diffusion VLA              │
└──────────────────────────────┬───────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
  [Partial Point Cloud]  [Language Instruction]  [Simba Completion]
    (N, 3)                   text                   (frozen)
        │                      │                      │
        │                      ▼                      │
        │              ┌────────────────┐             │
        │              │  TextEncoder   │             │
        │              │ (BiGRU, 256d)  │             │
        │              └───────┬────────┘             │
        │                      │                      │
        │                      │                      ▼
        │                      │              ┌──────────────┐
        │                      │              │ Simba Model  │
        │                      │              │ (Mamba+Diff)  │
        │                      │              └──────┬───────┘
        │                      │                     │
        ▼                      │                     ▼
  ┌──────────────┐             │           ┌──────────────────┐
  │ PointNet++   │             │           │   PointNet++     │
  │ + FiLM Cond  │◄────text───┘───text───►│   + FiLM Cond    │
  │ (partial)    │                         │   (completed)    │
  └──────┬───────┘                         └────────┬─────────┘
         │ 512d                                     │ 512d
         │                                          │
         └─────────────┬───────────────────┬────────┘
                       │                   │
                       ▼                   ▼
               ┌───────────────────────────────┐
               │    DualBranchEncoder Fusion    │
               │    partial_feat ⊕ comp_feat    │
               └───────────────┬───────────────┘
                               │ 512d
                               ▼
               ┌───────────────────────────────┐
               │     ObservationFusion         │
               │  visual + text (+ proprio)    │
               │  + TemporalAggregator (opt)   │
               └───────────────┬───────────────┘
                               │ 512d
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────┐  ┌──────────────┐  ┌──────────────┐
        │Classifier│  │ Diffusion    │  │Deterministic │
        │(optional)│  │ Action Head  │  │ Head (abla.) │
        │ softmax  │  │ DDPM/DDIM    │  │  MLP regress │
        └──────────┘  └──────────────┘  └──────────────┘
```

### 模块清单

| 模块 | 文件 | 功能 |
|------|------|------|
| PointNet++ 编码器 | `vla/encoder.py` | 层次化 3D 特征提取 (FPS + SA + MaxPool) |
| FiLM 条件化 | `vla/encoder.py` | 文本特征调制 3D 表征 (scale + shift) |
| 交叉注意力 | `vla/encoder.py` | 几何特征 attend 到文本语义 |
| 双分支编码 | `vla/encoder.py` | partial + completed 独立/共享编码 |
| 文本编码器 | `vla/model.py` | 双向 GRU + LayerNorm |
| 扩散动作头 | `vla/diffusion_policy.py` | 条件 DDPM + 1D U-Net 去噪器 |
| 确定性动作头 | `vla/diffusion_policy.py` | MLP 回归 baseline (消融用) |
| 时序聚合 | `vla/temporal.py` | Transformer + 可学习时序位置编码 |
| 本体感知编码 | `vla/temporal.py` | 关节/末端执行器状态编码 |
| 多模态融合 | `vla/temporal.py` | visual + text + proprio 统一表征 |
| 遮挡模拟 | `vla/occlusion.py` | 5 种遮挡模式 (视角/平面/随机/距离/扇区) |
| Simba 补全 | `vla/model.py` | 冻结的 Simba 模型，对称感知扩散补全 |
| 简单分词器 | `vla/tokenizer.py` | 正则分词 + BPE-free 轻量词表 |
| 数据加载 | `vla/dataset.py` | 支持操作/驾驶多场景标注格式 |

---

## 3. 核心模块详解

### 3.1 PointNet++ 3D 编码器 (`vla/encoder.py`)

**为什么选 PointNet++？**

- 对无序点集天然兼容，不需要体素化或投影
- 层次化集合抽象 (Set Abstraction) 同时捕获局部几何和全局结构
- 在实际机器人系统中广泛验证 (DP3, PerAct, Act3D)
- 计算开销远小于 Point Transformer V2/V3

**三级 Set Abstraction 层设计：**

```
输入: (B, N, 3)
  └─ SA1: FPS(512) + KNN(32) + MLP[3→64→64→128]    → (B, 512, 128)
      └─ SA2: FPS(128) + KNN(64) + MLP[131→128→128→256] → (B, 128, 256)
          └─ SA3: Global + MLP[259→256→512→512]      → (B, 512)  全局特征
```

每级 SA 层内部：
1. **Farthest Point Sampling (FPS)**: 采样中心点，保证空间均匀覆盖
2. **Ball Query / KNN**: 在中心点周围聚集局部邻域
3. **Local Coordinates**: 邻点坐标减去中心点 → 平移不变性
4. **Mini-PointNet**: shared MLP + MaxPool 提取局部特征

**FiLM 条件化 (Feature-wise Linear Modulation)：**

文本特征通过 MLP 生成 γ (scale) 和 β (shift)，对 3D 全局特征做仿射变换：

```
conditioned = global_3d * (1 + γ_text) + β_text
```

这让相同几何在不同指令下产生不同表征（例如 "pick the red mug" vs "push the red mug"）。

**Cross-Attention 层：**

在 FiLM 之上增加一层多头交叉注意力，让 3D 特征显式 attend 到文本 token，捕捉更细粒度的语义-几何对齐。

### 3.2 扩散动作头 (`vla/diffusion_policy.py`)

**为什么用 Diffusion Policy 而不是 MLP 回归？**

核心原因：**动作分布是多模态的**。

当指令是 "把杯子放到桌子上" 时，有效的放置位置是一个分布，不是一个点。MLP 回归只能输出均值（mode averaging），导致预测的动作在两个有效动作之间取平均，可能是一个无效动作。Diffusion Policy 通过从噪声出发迭代去噪，天然建模多模态分布。

**DDPM 前向/反向过程：**

前向（加噪）：$q(a_t | a_0) = \mathcal{N}(a_t; \sqrt{\bar\alpha_t} a_0, (1-\bar\alpha_t) I)$

反向（去噪）：$p_\theta(a_{t-1} | a_t, c) = \mathcal{N}(a_{t-1}; \mu_\theta(a_t, t, c), \sigma_t^2 I)$

其中 $c$ 是观测条件 (3D特征 + 文本特征 融合后的向量)。

**去噪器架构 (ActionDenoiserUNet)：**

```
输入: noisy_action (B, action_dim, horizon)
  │
  ├─ TimeEmbedding: sinusoidal → MLP → t_emb
  ├─ CondProjection: obs_feature → c_emb
  │
  ├─ Encoder:
  │   ├─ ResBlock(base_dim, t_emb, c_emb) + Downsample
  │   ├─ ResBlock(base_dim*2, t_emb, c_emb) + Downsample
  │   └─ ResBlock(base_dim*4, t_emb, c_emb) + Downsample
  │
  ├─ Bottleneck: ResBlock(base_dim*4, t_emb, c_emb)
  │
  └─ Decoder (with skip connections):
      ├─ Upsample + Cat(skip) + ResBlock
      ├─ Upsample + Cat(skip) + ResBlock
      └─ OutputProj → predicted_noise (B, action_dim, horizon)
```

每个 ResBlock 内部：
- Conv1d + GroupNorm + Mish 激活
- 时间嵌入通过 addition 注入（控制去噪步速率）
- 条件嵌入通过 FiLM 注入（控制什么动作该被去噪出来）

**Cosine Noise Schedule：**

比 linear schedule 更平滑，对低维动作空间 (7-DoF) 更友好。

**DDIM 加速采样：**

推理时不需要跑完 100 步 DDPM，用 DDIM 只需 10 步即可得到高质量动作，推理速度提升 10x。对实时性要求高的场景（自动驾驶）非常关键。

### 3.3 Simba 点云补全 (`models/Simba.py`)

Simba 是本项目的基础——一个对称感知的扩散点云补全模型。

**核心创新：**

1. **Mamba 序列建模**: 替代 Transformer 的二次复杂度，对长序列点云高效建模
2. **对称约束**: 预测 SE(3) 变换实现对称补全
3. **多级精修**: 3 个 Refiner 逐步上采样精修几何细节
4. **扩散去噪**: 在粗补全基础上做 diffusion denoising 消除噪声

**在 VLA 中的角色：**

Simba 以冻结模式工作，接收 partial point cloud，输出 completed point cloud。关键洞察：**补全不是为了视觉好看，而是为了降低动作决策的状态不确定性**。

当机器人看到一个被遮挡一半的杯子时：
- Partial only: 策略只看到半个杯子，不确定杯子的中心、朝向、完整尺寸
- Completed: Simba 恢复出完整杯子 → 策略可以精确规划抓取点

### 3.4 遮挡模拟 (`vla/occlusion.py`)

5 种遮挡模式覆盖不同的真实场景：

| 模式 | 真实对应场景 | 实现方式 |
|------|------------|---------|
| `viewpoint` | 单视角深度相机 | 按距离排序保留 closest k% |
| `planar` | 桌面/墙面遮挡 | 切割平面一侧移除 |
| `random` | 传感器噪声/丢包 | 均匀随机丢弃 |
| `distance` | 远距离 LiDAR 稀疏 | 按到参考点距离过滤 |
| `sector` | 扇形视角受限 | 按角度扇区移除 |

**Severity 参数**: 0 (无遮挡) → 1 (几乎全部遮挡)

**遮挡 sweep 实验**: `eval_vla.py --occlusion_sweep` 会在 [0, 0.1, 0.2, ..., 0.8] 的遮挡率下分别评估，生成性能退化曲线，直观展示 Simba 补全的价值。

### 3.5 时序聚合 (`vla/temporal.py`)

**为什么需要时序信息？**

单帧观测只能做反应式决策，无法推断：
- 物体的运动速度和方向
- 自车/机器人的当前动态状态
- 历史轨迹对未来规划的约束

**Temporal Transformer Aggregator：**
- 可学习的时序位置编码 (不是固定 sinusoidal)
- 带可学习聚合 token 的 Transformer Encoder
- 支持变长历史 (padding mask)

```
观测历史: [obs(t-3), obs(t-2), obs(t-1), obs(t)]
  → +时序位置编码
  → [AGG_TOKEN] + [obs(t-3), obs(t-2), obs(t-1), obs(t)]
  → TransformerEncoder
  → AGG_TOKEN 输出 = aggregated observation feature
```

---

## 4. 训练与评估流程

### 训练

```bash
# 基础 VLA (简单模型，快速验证)
python tools/train_vla.py --config cfgs/VLA_models/SimbaVLA.yaml

# DP3 风格扩散策略 + 操作场景
python tools/train_vla.py --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml

# DP3 风格扩散策略 + 自动驾驶场景
python tools/train_vla.py --config cfgs/VLA_models/SimbaVLA_DP3_driving.yaml

# 消融: 确定性 head (无扩散)
python tools/train_vla.py --config cfgs/VLA_models/SimbaVLA_ablation_deterministic.yaml
```

### 评估

```bash
# 标准评估
python tools/eval_vla.py \
  --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml \
  --checkpoint experiments/simba_vla_dp3_manipulation/best_model.pth

# 遮挡鲁棒性 sweep
python tools/eval_vla.py \
  --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml \
  --checkpoint experiments/simba_vla_dp3_manipulation/best_model.pth \
  --occlusion_sweep \
  --occlusion_method viewpoint \
  --output_json results/occlusion_sweep.json

# 按场景分解指标
python tools/eval_vla.py \
  --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml \
  --checkpoint experiments/simba_vla_dp3_manipulation/best_model.pth \
  --per_scenario
```

### 推理

```bash
python tools/infer_vla.py \
  --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml \
  --checkpoint experiments/simba_vla_dp3_manipulation/best_model.pth \
  --point_cloud demo/mug_partial.pcd \
  --instruction "pick up the mug from the table" \
  --save_completed_points results/completed.npy
```

---

## 5. 配置系统与多场景

### 场景切换

项目通过 YAML 配置文件无缝切换场景，不需要改代码：

| 配置文件 | 场景 | Action Space | 关键区别 |
|---------|------|-------------|---------|
| `SimbaVLA.yaml` | 基础 demo | 4D (简单) | 简单 MLP 模型 |
| `SimbaVLA_DP3_manipulation.yaml` | 操作 | 7D (pos+rot+gripper) | PointNet++ + Diffusion |
| `SimbaVLA_DP3_driving.yaml` | 驾驶 | Waypoints (4步×3D) | LiDAR 补全 + 轨迹预测 |
| `SimbaVLA_ablation_deterministic.yaml` | 消融 | 7D | MLP head, 无扩散 |

### Action Space 设计

**操作场景 (7-DoF)：**
```
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
  位置增量    旋转增量 (欧拉角)      0=开 1=闭
```

**驾驶场景 (Waypoint)：**
```
[[x1,y1,z1], [x2,y2,z2], [x3,y3,z3], [x4,y4,z4]]
  未来 4 个时间步的世界坐标位置
```

**驾驶场景 (Control)：**
```
[throttle, steering, brake]
   油门      方向盘     刹车
```

---

## 6. 实验设计与 Ablation

### 核心实验矩阵

| 实验 | 目标 | 变量 |
|------|------|------|
| **Exp1**: Input Ablation | 补全是否有用 | partial / completed / partial+completed |
| **Exp2**: Occlusion Sweep | 补全在高遮挡下更有用 | severity ∈ [0, 0.1, ..., 0.8] |
| **Exp3**: Action Head | 扩散 vs 确定性 | diffusion / deterministic |
| **Exp4**: Encoder | PointNet++ vs MLP | use_pointnetpp true/false |
| **Exp5**: 场景迁移 | 模型泛化性 | manipulation / driving cross-eval |

### 预期结论

1. **补全提升**: partial+completed > completed > partial，差距在高遮挡下放大
2. **扩散优势**: diffusion > deterministic，尤其在多模态动作场景
3. **编码器**: PointNet++ > simple MLP，因为层次化特征更适合 3D policy
4. **鲁棒性**: 补全带来的收益在 severity > 0.4 时最显著

### 指标体系

| 指标 | 场景 | 含义 |
|------|------|------|
| Accuracy | 操作/驾驶 | 离散动作类别预测准确率 |
| Action MAE | 操作 | 连续动作向量的平均绝对误差 |
| ADE (Average Displacement Error) | 驾驶 | 轨迹平均偏差 |
| FDE (Final Displacement Error) | 驾驶 | 终点偏差 |
| CD (Chamfer Distance) | 补全 | Simba 补全质量 |
| F-score@0.01 | 补全 | 补全完整度 |

---

## 7. 简历写法建议

### 通用版本

> **Completion-Augmented 3D Diffusion Policy for Occlusion-Robust VLA**
>
> - 基于 Simba 搭建 completion-augmented 3D diffusion VLA pipeline，将对称感知点云补全（Mamba+Diffusion）与 DP3 风格扩散策略串联，支持 partial point cloud + language instruction 到 7-DoF action 的端到端建模
> - 设计 PointNet++ + FiLM conditioning + cross-attention 的 3D-language 编码器，以及条件 DDPM 动作去噪器（cosine schedule + 1D U-Net），支持 DDIM 10 步加速采样
> - 实现 5 种遮挡模拟模式的 robustness benchmark，对比 partial-only / completed-only / dual-branch 三种输入范式，验证几何补全在高遮挡率下对策略鲁棒性的系统性贡献

### 偏具身 / 机器人

> **Occlusion-Robust Robotic Manipulation via Point Cloud Completion + 3D Diffusion Policy**
>
> - 提出 completion-augmented manipulation pipeline: Simba 恢复遮挡几何 → PointNet++ 编码 partial+completed 点云 → 双向 GRU 编码语言指令 → DDPM 条件动作生成
> - 在 tabletop manipulation 场景验证，支持 7-DoF action space (position + rotation + gripper)，扩散动作头在多模态抓取任务上优于 MLP 回归 baseline
> - 遮挡 sweep 实验证明：当遮挡率 > 40% 时，Simba 补全使策略准确率提升显著，验证了 3D state completion 对 embodied decision-making 的价值

### 偏自动驾驶

> **Completion-Augmented LiDAR VLA for Command-Conditioned 3D Trajectory Prediction**
>
> - 将稀疏 LiDAR 点云补全（Simba，基于 Mamba + 对称扩散）接入 command-conditioned 轨迹预测 pipeline，支持 navigation instruction → future waypoint regression
> - 设计 PointNet++ + 扩散 action head 架构，以 ADE/FDE 为指标在多种遮挡条件下评估 LiDAR 稀疏性对规划性能的影响
> - 实验表明补全后的密集点云显著改善轨迹预测精度，尤其在远距离/遮挡严重的场景下

---

## 8. 面试深度 Q&A

### A. 项目整体设计 (10 题)

**Q1: 为什么要先做点云补全，再做 VLA？直接 end-to-end 预测不行吗？**

A: 直接 end-to-end 预测在完美观测下是可以的，但在遮挡/稀疏场景下，partial observation 的状态信息不足以支撑可靠的动作决策。几何补全的本质是 **状态恢复**——降低观测到状态映射中的不确定性。这遵循了经典 POMDP (Partially Observable MDP) 的思路：先做 belief update (补全 ≈ 恢复 full state belief)，再在更完整的状态空间上做策略学习。实验上，这种两阶段方法在 occlusion sweep 下展现出明显更好的鲁棒性。

**Q2: 这个项目和世界模型有什么关系？**

A: 世界模型的核心是 "从当前观测预测/恢复状态 → 在状态空间做规划"。这个项目做的正是：(1) Simba 从 partial observation 恢复完整 3D 状态 = 世界模型的 state estimation 部分；(2) Diffusion Policy 在恢复的状态上做 conditional action generation = 世界模型的 policy/planning 部分。和 Dreamer / TDMPC 不同的是我们在 3D 几何空间做 state recovery 而不是在 latent space，这提供了可解释性。后续扩展方向是加入 temporal dynamics 做多步状态预测 → 真正的 3D world model。

**Q3: 为什么不直接用 CLIP/BERT 做文本编码？**

A: 对于操作指令和导航命令这种 closed-domain 短文本，双向 GRU 足够了。CLIP 的优势在于 open-vocabulary zero-shot 能力，但它的文本编码器是为自然图片描述优化的，对 "move gripper 10cm left" 这种指令级文本并不特别适合。更重要的是，GRU 编码器可以和 3D 编码器一起 end-to-end 训练，FiLM conditioning 的梯度可以直接回传到文本表征。如果要扩展到 open-vocabulary 场景，可以替换为 frozen CLIP text encoder + 可学习 projection。

**Q4: Simba 为什么用冻结模式？不微调会不会损失性能？**

A: 用冻结模式有三个原因：(1) Simba 在大规模点云补全数据上预训练，已经捕获了丰富的几何先验，微调可能导致灾难性遗忘；(2) 分离训练使得 VLA 的训练更稳定，不需要平衡补全损失和动作损失；(3) 冻结 Simba 让实验更干净——VLA 性能的提升完全来自 "更好的输入" 而不是 "补全模型变好了"。如果有足够多的 VLA 训练数据，可以用很小的 learning rate 做 fine-tuning。

**Q5: 动作空间为什么选 7-DoF？**

A: 7-DoF = 3D position (Δx, Δy, Δz) + 3D rotation (Δroll, Δpitch, Δyaw) + gripper state (0/1)。这是机器人操作中最标准的末端执行器控制空间。RLBench/ManiSkill/BridgeData 等主流 benchmark 都用这个空间。对于自动驾驶场景则切换成 waypoint (未来 T 步的 3D 坐标) 或 control (throttle/steering/brake)。

**Q6: 为什么用 PointNet++ 而不是 Point Transformer？**

A: PointNet++ 在 3D policy learning 中是经过充分验证的选择 (DP3, PerAct)，它的 Set Abstraction 层对不规则、稀疏、有噪声的真实点云更鲁棒。Point Transformer V2/V3 虽然在 3D 分割 benchmark 上更强，但 (1) 计算开销更大，(2) 在 policy learning 场景下并未展现出决定性优势，(3) 需要更多的数据才能发挥长距离注意力的优势。在资源有限时，PointNet++ 是 cost-effective 的选择。

**Q7: Diffusion Policy 的推理速度怎么样？能实时吗？**

A: 标准 DDPM 100 步采样确实慢。我们实现了 DDIM 采样，只需 10 步就能生成高质量动作，推理延迟可以降到几十 ms 级别。进一步加速可以用 Consistency Model (一步生成) 或 Rectified Flow (线性采样路径)。对于自动驾驶 100ms 控制周期，DDIM 10 步已经可以满足。

**Q8: 双分支编码 (partial + completed) 会不会导致参数翻倍？**

A: 是的，不共享时参数量接近翻倍。但 PointNet++ 本身很轻量 (几M参数)，翻倍后仍然远小于 LLM-based VLA (几B参数)。也可以用 `share_backbone=True` 让两个分支共享权重，只在最后的 FiLM conditioning 层分开——这样参数量几乎不增加，但仍然保留了区分 partial/completed 输入的能力。实验上需要 ablation 哪种更好。

**Q9: 这个项目能部署到真实机器人上吗？**

A: 架构上完全可以。部署路径是：(1) 用 RGBD 相机获取点云 → (2) Simba 补全 → (3) diffusion policy 生成动作 → (4) 发送到机器人控制器。实际部署需要解决：传感器标定、点云预处理 pipeline、推理延迟优化 (TensorRT / ONNX)、安全约束。这个项目目前聚焦在 pipeline 验证而非部署工程。

**Q10: 和 RT-2 / Octo / OpenVLA 这些大模型 VLA 比，你的优势是什么？**

A: 大模型 VLA (RT-2 / OpenVLA) 的策略压在 VLM 的 token 预测上，它们擅长 open-vocabulary 泛化。我们的优势在于：(1) **3D-native**: 直接在点云空间操作，不丢失 3D 几何信息；(2) **遮挡处理**: 有显式的几何补全阶段，这是大模型 VLA 不具备的结构先验；(3) **轻量**: 可以在单卡上训练，不需要大规模数据和算力。劣势是开放词汇能力弱。两者是互补的——大模型 VLA 做高层指令理解，我们做精确的 3D 几何感知和动作生成。

---

### B. 扩散模型深度 (10 题)

**Q11: 为什么选 cosine schedule 而不是 linear？**

A: 对低维动作空间 (7D)，linear schedule 的噪声增长太快，导致前几步加噪过度、信号被淹没。Cosine schedule (Nichol & Dhariwal, 2021) 的变化更平滑，在训练初期保留更多信号，让模型更容易学到细粒度的动作调整。实验上 cosine schedule 在动作预测任务上收敛更快、结果更好。

**Q12: DDPM 训练的 loss 是什么？直觉是什么？**

A: 训练 loss 是 **noise prediction MSE**：采样随机时间步 t，给 ground truth 动作加噪声 ε，让网络预测加上去的噪声 ε_θ。等价于让网络学会 "这个噪声动作里，哪些是信号，哪些是噪声"。直觉上，去噪器学到的是动作分布的 **score function** (对数概率的梯度)，推理时沿梯度方向走，从噪声走到高概率的真实动作。

**Q13: DDIM 和 DDPM 的本质区别是什么？**

A: DDPM 的每步去噪是 stochastic 的 (加随机噪声)，必须走完所有 T 步。DDIM 的去噪是 deterministic 的 (eta=0 时)，可以跳步。本质上 DDIM 把 DDPM 的马尔可夫链重新参数化成了一个 non-Markovian 过程，允许用更少的步数到达相同的终点。eta=0 时从同一个初始噪声生成同一个动作 → 可复现性好。

**Q14: 条件注入为什么用 FiLM 而不是 concatenation？**

A: FiLM (Feature-wise Linear Modulation) 对每个 feature channel 做独立的 scale+shift，让条件信息 **调制** 特征而不是简单拼接。Concatenation 需要网络自己学出什么时候该关注条件，FiLM 直接提供了 per-channel 的门控信号，在条件生成任务中更高效。GAN / conditional diffusion 文献中 FiLM 几乎是标准做法。

**Q15: 为什么用 1D U-Net 做去噪器，而不是 MLP？**

A: 当 action horizon > 1 (多步预测) 时，动作序列有时序结构。1D U-Net 用卷积捕获局部时序模式 (如加速/转向连贯性)，用 skip connection 保留多尺度信息，用 downsample/upsample 编码长短期依赖。MLP 把所有时间步拍平，丢失了局部连贯性。对 horizon=1 的场景，1D U-Net 退化成简单的 residual blocks，不会引入额外开销。

**Q16: Diffusion Policy 怎么处理动作约束 (joint limits, workspace bounds)？**

A: 当前实现用 `clip_sample=True` 把去噪结果 clamp 到 [-1, 1]，配合动作 normalization 到 [-1, 1] 范围。更精细的做法包括：(1) 在训练时对 ground truth normalize 到约束范围内；(2) 在推理时加投影 (project onto feasible set)；(3) 用 guided diffusion 把约束作为 energy function 加到去噪过程中。

**Q17: 多模态动作分布的具体例子？**

A: "把杯子放到桌子上" → 桌面上有多个合法放置区域，形成多峰分布。"绕过障碍物到达目标" → 可以左绕或右绕，形成双峰。"抓取一个圆柱体" → 可以从任意角度接近，形成环形分布。MLP 回归在这些情况下会输出多峰的均值 (可能是非法的中间状态)，diffusion 采样则会 sample 到其中一个峰。

**Q18: 你的去噪器有多少参数？和 Diffusion Policy 论文比呢？**

A: base_dim=128, dim_mults=(1,2,4) → ResBlock 约 128/256/512 channels → 去噪器约 2-5M 参数。Diffusion Policy 论文用 256 base dim + 更多层，约 10-20M。我们故意做得更轻量，因为 VLA 的重点不仅在动作生成，还在 3D 编码和补全。可以根据任务复杂度调节 base_dim。

**Q19: 能不能用 Consistency Model / Flow Matching 替代 DDPM？**

A: 完全可以，而且是自然的升级方向。Consistency Model 只需 1 步生成 (Consistency Distillation 或 Consistency Training)，延迟极低。Flow Matching / Rectified Flow 用线性 ODE 路径，训练更稳定。PointFlowMatch (2024) 就是用 Flow Matching 做 3D policy 的。架构上只需要替换 DiffusionActionHead 内部的采样逻辑，其他模块不变。

**Q20: diffusion loss 和 classification loss 的 multi-task 平衡怎么做？**

A: 当前用 `cls_loss_weight` 和 `reg_loss_weight` 手动设置。更好的做法是 uncertainty weighting (Kendall et al., 2018)：为每个 loss 学一个 precision (log variance) 参数，自动平衡不同 loss 的量级。或者用 GradNorm / MGDA 做显式的梯度平衡。在我们的场景中，classification loss 主要提供辅助信号 (粗粒度动作类型)，权重通常设为 0.5。

---

### C. 3D 感知与编码 (10 题)

**Q21: FPS 采样为什么比随机采样好？**

A: FPS (Farthest Point Sampling) 保证采样点在空间中均匀分布，不会出现 "一团点扎堆，远处空白" 的情况。对于稀疏/遮挡点云尤其重要——随机采样可能在已经稀疏的区域进一步丢失信息，FPS 则尽量保留空间覆盖率。

**Q22: Ball Query 和 KNN 的区别？什么时候用哪个？**

A: Ball Query 用固定半径搜索邻居，邻居点数可变；KNN 用固定 K 搜索最近邻，半径可变。Ball Query 在密度均匀时更好 (一致的感受野)，KNN 在密度不均匀时更鲁棒 (稀疏区域也能找到 K 个邻居)。对于遮挡后的稀疏点云，KNN 更合适，所以我们默认 `use_knn=True`。

**Q23: Set Abstraction 的 local coordinates 为什么重要？**

A: `grouped_xyz -= center_xyz` 让每个局部邻域的特征相对于中心点表示。这赋予了网络 **平移不变性**——同一个几何结构出现在空间的不同位置，会产生相同的局部特征。这对机器人操作至关重要（杯子在桌子左边和右边应该产生相同的局部特征）。

**Q24: Cross-Attention 和 FiLM 为什么同时使用？不冗余吗？**

A: FiLM 是 **全局调制**——把整个文本语义浓缩成 scale/shift 施加在所有 3D 特征上。Cross-Attention 是 **局部对齐**——让 3D 特征的不同部分选择性地关注文本的不同部分。两者互补：FiLM 提供全局先验 ("这是一个抓取任务")，Cross-Attention 提供细粒度对齐 ("left side" → 关注左侧几何)。在实践中，单用 FiLM 在简单指令上够用，加 Cross-Attention 在复杂指令上有提升。

**Q25: 多分支编码的 partial+completed concat 和 attention fusion 哪个好？**

A: Concat + MLP 是简单有效的选择，attention fusion 能让模型动态选择两个分支的信息。对我们的场景，concat 更稳定：因为 partial 和 completed 的信息是互补的 (partial 保留真实传感器特征，completed 提供完整几何)，直接融合符合直觉。如果两个分支信息重叠度高 (例如 completion quality 很好时)，attention 可以学到自适应权重。

**Q26: 点云的 normalize 具体做了什么？为什么需要？**

A: (1) 中心化：减去质心，消除绝对位置影响；(2) 缩放：除以最大范数，映射到单位球。需要的原因：不同物体的尺度差异很大 (杯子 10cm vs 桌子 1m)，不 normalize 的话网络需要同时学 scale-invariant 的特征和 scale-specific 的动作映射，增加学习难度。注意推理时需要用保存的 center 和 scale 做反归一化。

**Q27: 你的 3D 编码器能处理不同点数的输入吗？**

A: 可以。FPS + KNN 操作的输入点数是灵活的。但为了 batch 处理，我们在 dataset 层做了 resampling 到固定 num_points。如果要支持变长输入，可以用 padding + mask 的方式。PointNet++ 的天然优势就是对点数不敏感——SA 层会逐步下采样到固定点数 (512→128→1)。

**Q28: 和体素化方法 (MinkowskiEngine / 3D sparse conv) 比，点云方法的优劣？**

A: 点云方法 (PointNet++): 直接处理不规则点集，不丢失定位精度，对稀疏数据高效。劣势：KNN/FPS 的 GPU 实现在大规模场景 (10万+点) 下慢。3D sparse conv (MinkowskiEngine): 对大规模场景 (完整 LiDAR sweep) 更高效，kernel-based 操作在固定网格上天然支持 convolution。劣势：体素化会丢失亚体素精度，需要选体素分辨率。对我们的场景 (单物体/小场景, 2k-8k点)，PointNet++ 足够；如果扩展到 KITTI 全场景补全 → 3D sparse conv 更合适。

**Q29: 从编码器取什么特征给动作头？global feature 还是 per-point feature？**

A: 当前用 global feature (SA3 输出的单向量)。这对分类和简单回归足够。如果动作需要空间定位 (例如 "在这个点抓取")，需要 per-point feature + 空间投票。DP3 paper 做了对比，global feature 在 manipulation 上已经很好。如果后续做 spatial action prediction (像素级/点级的动作预测)，需要保留 multi-scale feature 做 decoder。

**Q30: 如果输入点云全是噪声 (完全随机)，模型会怎样？**

A: 理论上 FPS 仍然会均匀采样 "噪声点"，SA 层会提取 "噪声几何特征"，但这些特征对 policy 没有信号。模型应该退化到 text-only 模式——只依赖语言指令做决策。可以在训练时加一些 fully corrupted 样本作为 negative augmentation，让模型学会在观测无效时 fallback 到安全动作 (如 stop / hold)。

---

### D. 具身智能 / 机器人 (10 题)

**Q31: 你的模型和 RLBench/CALVIN/ManiSkill 怎么对接？**

A: 这些 benchmark 提供 (observation, language instruction, action) 的配对数据。对接路径：(1) 从 RGBD observation 通过 depth unprojection 得到点云；(2) 用 camera extrinsic 合并多视角点云为 workspace point cloud；(3) 对 partial observation 可以直接传 Simba；(4) action 通常是 7-DoF delta EE pose + gripper。需要写一个 dataset adapter 把 benchmark 格式转成我们的 JSON annotation 格式。

**Q32: 真实机器人上怎么获取 partial point cloud？**

A: (1) **RGBD 相机** (RealSense, Kinect): depth map 反投影得到有序点云，天然是单视角 partial；(2) **LiDAR** (Livox, Ouster): 直接输出 3D 点云，远距离/遮挡区域稀疏；(3) **多相机融合**: 多个 RGBD 视角的点云在世界坐标系下合并，仍然有死角区域。所有这些都会产生 partial observation，都能 feed 给 Simba 补全。

**Q33: 遮挡 robustness 在真实 manipulation 中有多重要？**

A: 非常重要。桌面操作中：(1) 物体被其他物体遮挡 (堆叠场景、杂乱桌面)；(2) 自身遮挡 (杯子把手在背面)；(3) 机器人手臂遮挡 (end-effector 挡住观测)。据 RLBench 数据集统计，约 30-50% 的任务实例存在显著遮挡。如果策略只在完美观测上训练，部署时会 catastrophic fail。

**Q34: Diffusion Policy 在操作任务中的成功率通常是多少？**

A: DP3 论文 (Ke et al., RSS 2024) 在 MetaWorld/Adroit 上达到 70-90% success rate。Diffusion Policy (Chi et al.) 在 real robot pushing 上达到 ~80%。在 simulation (RLBench) 上对简单任务 (reach, push) 接近 100%，复杂任务 (stack, insert) 60-80%。我们的补全-增强版本预期在遮挡场景下比 vanilla DP3 有 5-15% 的提升。

**Q35: 你的 temporal aggregation 和 action chunking 有什么区别？**

A: Action Chunking (ACT, Zhao et al.) 是在 **输出端** 一次预测多步动作 (我们的 action_horizon > 1)。Temporal Aggregation 是在 **输入端** 聚合多帧观测。两者是正交的，可以同时使用：多帧观测 → temporal transformer → 状态表征 → diffusion head → 多步动作 chunk。

**Q36: 怎么处理长期任务 (multi-step, 需要多次 pick-and-place)？**

A: 当前模型是 **reactive policy**: 每一步给一个观测和指令，预测下一步动作。长期任务需要 (1) 高层规划器分解成子任务序列 (e.g., LLM planner)；(2) 每个子任务对应一条语言指令；(3) 我们的模型执行每个子任务的低层控制。这遵循 SayCan / Code as Policies 的 hierarchical 范式。

**Q37: 机器人 sim-to-real transfer 的挑战？**

A: (1) **视觉 gap**: simulation 的点云和真实传感器差别大 (噪声、分辨率、反射特性)；(2) **物理 gap**: 仿真器的接触/摩擦不够真实；(3) **domain randomization**: 训练时随机化光照、纹理、物理参数。我们的遮挡模拟 (`occlusion.py`) 本身就是一种 domain randomization，可以缩小 sim-to-real gap。

**Q38: 你的 gripper action 是离散 (open/close) 还是连续的？**

A: 配置中是连续值 [0, 1]，但实际部署在 parallel jaw gripper 时通常二值化 (threshold at 0.5)。连续建模的好处是可以表示不同抓力，对柔性物体有用。

**Q39: 如果有多个可交互物体，模型怎么知道操作哪个？**

A: 语言指令负责指定目标 ("pick up the **red mug**")。Cross-Attention 机制让 3D 特征 attend 到 "red mug" 对应的文本 token，从而关注特定物体的几何区域。如果场景中有多个红色杯子，需要额外的 grounding 机制 (如 3D referring expression comprehension)。

**Q40: 这个方法能做双臂协作吗？**

A: 架构上可以，把 action_dim 从 7 扩展到 14 (双臂各 7-DoF)。更好的做法是双臂 factorize: master arm 和 slave arm 各一个 policy，通过 shared observation feature 协调。当前代码不直接支持，但加一个 multi-agent wrapper 就可以。

---

### E. 自动驾驶 / 规划 (10 题)

**Q41: 你的 driving 场景和 UniAD/VAD/DriveVLM 对比？**

A: UniAD/VAD 是 **end-to-end sensor → planning** pipeline，输入原始图像/多视角，输出轨迹。我们的关注点不同：聚焦在 **LiDAR 稀疏/遮挡条件下的补全对规划的辅助**。定位上更像 3D LiDAR perception 模块里的一个组件，可以插入到 UniAD 这样的大 pipeline 中。

**Q42: Waypoint prediction 和 trajectory planning 有什么区别？**

A: Waypoint prediction 预测离散的未来位置点 (our approach)，trajectory planning 输出连续的参数化轨迹 (如样条曲线)。Waypoint 更简单直接，但可能不保证运动学可行性。在实际系统中通常 waypoint prediction → trajectory optimization (保证曲率、速度约束) → control command。

**Q43: ADE/FDE 指标怎么算？有什么局限？**

A: ADE = 所有时间步预测与 GT 位移的平均 L2；FDE = 最后一个时间步的 L2。局限：(1) 对 time alignment 敏感——即使轨迹形状正确但速度不同，ADE 也会很大；(2) 不考虑可行性 (预测的轨迹可能穿墙)；(3) 多模态评估需要 minADE (K 条预测取最小)。

**Q44: 为什么自动驾驶需要 LiDAR 补全？**

A: (1) 远距离物体只有几个点 (200m 外行人<5点)，检测困难；(2) 遮挡区域 (被前车挡住) 完全没有返回点；(3) 传感器故障导致丢帧。补全后的密集点云显著提升下游检测 (AP+5-10%) 和规划 (ADE↓) 性能。KITTI / nuScenes 数据集中有专门的 LiDAR completion benchmark。

**Q45: command-conditioned planning 是什么意思？**

A: 不同于 unconditional trajectory prediction (预测未来轨迹分布)，command-conditioned planning 根据语言指令或导航命令 ("turn left", "change lane") 生成对应的特定轨迹。这正是我们项目做的事——语言指令作为条件，生成对应的动作/轨迹。

**Q46: 你的 driving config 里 action_horizon=4 是什么意思？**

A: 预测未来 4 个时间步的 waypoint，每个 waypoint 是 3D 坐标 (x, y, z)。所以 diffusion head 的输出是 (B, 3, 4)，reshape 后是 (B, 4, 3) = 4 个 3D waypoint。时间步间隔取决于数据标注的频率 (比如每 0.5s 一个 waypoint → 预测 2s 内的轨迹)。

**Q47: 怎么把 camera + LiDAR 的多模态融合加进来？**

A: 典型做法: (1) BEVFusion: 把 camera feature lift 到 BEV，和 LiDAR BEV concat；(2) 我们的框架可以加一个 image branch: 用 ResNet/ViT 编码图像 → 投影到和点云特征同维度 → 在 ObservationFusion 层 concat。架构上只需在 ObservationFusion 加一个 visual_image_dim 输入。

**Q48: diffusion policy 在 driving 场景的优势？**

A: 驾驶轨迹天然是多模态的——"continue straight" 在十字路口可以直行、左转、右转。Diffusion 可以 (1) 给定 "turn left" 命令 → 采样到左转轨迹；(2) 不给命令 → 采样到所有可能轨迹的分布。这比单一 MLP 回归输出一条平均轨迹有用得多。

**Q49: 你的框架能做 closed-loop simulation 吗？**

A: 当前是 open-loop evaluation (预测一次完整轨迹)。Closed-loop 需要把模型嵌入仿真器 (CARLA / nuPlan)，每步预测 → 执行 → 获取新观测 → 再预测。我们的模型支持这个 (每次传入当前点云 + 指令)，但需要仿真器集成，这是工程工作。

**Q50: 自动驾驶公司面试可能追问什么？**

A: (1) BEV 表示和你的点云表示的优劣对比；(2) 你的模型在 real-time 要求 (10Hz) 下能不能达标；(3) 怎么处理 safety-critical cases (突然出现的行人)；(4) 和 rule-based planner 的集成方式；(5) 传感器融合策略。准备好这些方向的回答。

---

### F. 世界模型 / 前沿技术 (10+ 题)

**Q51: 你理解的 "世界模型" 是什么？**

A: 世界模型是一个能预测 "如果我做了某个动作，世界状态会怎么变化" 的模型。形式化地，学习 p(s_{t+1} | s_t, a_t) (状态转移动态)。可以在像素空间 (video prediction)、语义空间 (scene graph dynamics)、或 latent space (Dreamer, TDMPC) 做。我们的项目中，Simba 的几何补全可以理解为对当前世界状态的恢复 (state estimation 部分)，后续扩展可以加上 temporal dynamics。

**Q52: Dreamer / TDMPC 和你的方法有什么区别？**

A: Dreamer / TDMPC 在 **latent space** 做 imagination (想象未来状态再做规划)。我们在 **3D geometric space** 做 state recovery (恢复当前被遮挡的几何)。区别：(1) 我们不预测未来状态变化，只恢复当前被遮挡的部分；(2) 我们的补全是可解释的 (输出真实点云)，latent world model 不可解释；(3) 扩展到 temporal dynamics 后两者可以统一。

**Q53: 怎么从你的项目扩展成真正的 3D world model？**

A: 三步：(1) 给 Simba 加上 temporal input → 从多帧部分观测预测当前完整状态 (4D completion)；(2) 加上 dynamics model → 给定当前完整状态和动作，预测下一时刻的状态；(3) 用预测的未来状态做 planning (model predictive control 或 tree search)。本质上是在 3D 点云空间做 Dreamer。

**Q54: 你的 "几何补全 = 隐式世界模型" 这个说法严谨吗？**

A: 不完全严谨。严格的世界模型需要 temporal dynamics (预测未来)。但 state estimation 是世界模型的核心前提——你不能预测一个你都不知道当前状态的世界会怎么变。所以更准确的说法是：**几何补全解决了 3D world model 的状态初始化问题**。

**Q55: Sora / Genie / UniSim 这些 video world model 和你的 3D approach 比？**

A: Video world model 在 2D 像素空间做 imagination，视觉效果好但缺乏 3D 物理一致性 (穿透、漂浮物体)。3D approach 天然有几何一致性和物理可行性约束。两者的融合趋势是 **3D-aware video generation** (如 4D Gaussian + dynamics)。我们的 3D completion 可以为 video world model 提供结构约束。

**Q56: 你怎么看 VLA 的未来发展方向？**

A: 三个方向：(1) **Scale**: 更大的预训练数据和模型 (OpenVLA, RT-X 路线)；(2) **3D-native**: 在 3D 表征而非 2D 图像上做 VLA (DP3, 我们的方向)；(3) **World Model integration**: VLA 不只是 perception → action，而是 perception → world model → imagination → planning → action。长期来看，(2) 和 (3) 会融合——在 3D 世界模型中做 language-conditioned planning。

---

## 9. 快速上手

```bash
# 训练基础模型 (无需 Simba 权重)
python tools/train_vla.py --config cfgs/VLA_models/SimbaVLA.yaml

# 训练 DP3 风格模型 (需要 Simba 权重开启补全)
python tools/train_vla.py --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml

# 评估 + 遮挡 sweep
python tools/eval_vla.py \
  --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml \
  --checkpoint experiments/simba_vla_dp3_manipulation/best_model.pth \
  --occlusion_sweep --output_json results/sweep.json

# 单样本推理
python tools/infer_vla.py \
  --config cfgs/VLA_models/SimbaVLA_DP3_manipulation.yaml \
  --checkpoint experiments/simba_vla_dp3_manipulation/best_model.pth \
  --point_cloud demo/mug_partial.pcd \
  --instruction "pick up the mug"
```

开启 Simba 补全：编辑 YAML 中 `model.use_completion: true` 并设置 `model.completion.checkpoint_path`。

---

## 10. 与前沿工作的对比定位

| 方法 | 输入 | 动作生成 | 3D 表征 | 补全 | 开源 |
|------|------|---------|---------|------|------|
| **RT-2** (Google, 2023) | RGB + text | VLM token prediction | 无 | 无 | 否 |
| **Octo** (UC Berkeley, 2024) | RGB + text | diffusion | 无 | 无 | 是 |
| **OpenVLA** (Stanford, 2024) | RGB + text | LLM token prediction | 无 | 无 | 是 |
| **DP3** (Ke et al., 2024) | Point cloud | diffusion | PointNet++ | 无 | 是 |
| **PerAct** (Shridhar et al., 2023) | Voxel grid | discrete | 3D voxel | 无 | 是 |
| **3D-VLA** (Zhen et al., 2024) | Multi-view + text | LLM-based | 3D tokens | 无 | 是 |
| **Ours** | Point cloud + text | diffusion | PointNet++ | **Simba** | 是 |

**我们的独特定位**: 唯一同时具备 **显式 3D 几何补全** + **diffusion action generation** + **language conditioning** 的 VLA pipeline。
