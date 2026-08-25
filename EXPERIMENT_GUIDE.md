# FedTaskPrompt 实验步骤手册

本文档说明如何从零开始运行 FedTaskPrompt 实验原型。所有命令默认在项目根目录
`fedtaskprompt/` 下执行。

本项目是算法验证原型，不模拟真实跨机构网络。单次实验仅创建一个冻结的语言模型
backbone，所有客户端在同一 Python 进程中顺序复用该模型。两到三张 GPU 应用于并行运行
不同随机种子或不同实验设置，而不是在单次实验中启用 DDP。

## 1. 当前实现状态

在正式运行前，应先区分已经接通的功能与仍待接入的功能。

| 功能 | 当前状态 | 说明 |
| --- | --- | --- |
| 任务编码向量余弦聚类 | 已接通 | 不再使用字段支持度离散匹配 |
| 全局唯一坐标生成器 | 已接通 | 所有任务簇共同更新一个 $G_\theta$ |
| 支持集适应与查询集反馈 | 已接通 | 支持一阶、坐标二阶和完整二阶模式 |
| Prompt 子空间与正交本地残差 | 已接通 | 包含投影与 basis 维护 |
| checkpoint 保存、恢复与独立评测 | 已接通 | 保存服务器、客户端和随机数状态 |
| 冷启动语义实验 | 已接通 | 当前为任务描述可见的传导式协议 |
| 经验迁移矩阵 | 已接通 | 使用 Local Prompt 构造两两迁移增益 |
| 部分消融实验 | 已接通 | 仅运行能够被当前配置严格表达的消融 |
| 五种基线算法类 | 已实现，未接入统一训练入口 | 暂不能直接生成完整基线对比表 |
| Accuracy、F1、ROUGE 等生成指标 | 指标函数已实现，未接入 evaluator | 当前统一训练入口主要输出 test loss |
| 通信量与运行效率统计 | 统计组件已实现，未接入 simulator 日志 | 当前训练输出不包含完整效率汇总 |
| basis drift 任务交换 | 只有配置，尚未接入 simulator | 不能用于正式漂移实验结论 |

因此，当前代码可以验证核心优化链路、语义分组、全局生成器更新、冷启动和 loss 级别结果，
但在基线统一入口和任务生成指标接通前，不应直接据此生成论文最终对比表。

## 2. 环境与硬件要求

### 2.1 推荐环境

- Linux；
- Python 3.11–3.13；
- 支持 BF16 的 NVIDIA GPU；
- CUDA 与 PyTorch 版本相互兼容；
- 可以访问 Hugging Face，或已经准备好离线模型和数据缓存；
- 使用 `uv` 管理依赖。

单个实验默认只使用一张 GPU。建议先用一张 GPU 完成小规模验证，再使用两到三张 GPU
并行运行独立随机种子。

### 2.2 检查 GPU

```bash
nvidia-smi
```

确认目标 GPU 可见：

```bash
CUDA_VISIBLE_DEVICES=0 nvidia-smi
```

### 2.3 安装 uv

如果系统尚未安装 `uv`，可按所在环境的统一软件管理方式安装。安装后检查：

```bash
uv --version
```

## 3. 创建环境并安装依赖

进入项目根目录：

```bash
cd /path/to/fedtaskprompt
```

生成并保留依赖锁文件：

```bash
uv lock
```

安装运行依赖和开发测试依赖：

```bash
uv sync --group dev
```

检查关键依赖：

```bash
uv run python -c "import torch, transformers, datasets, hydra; print(torch.__version__)"
```

检查 CUDA：

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

如果这里返回 `False`，不要直接开始 GPU 实验，应先修复 PyTorch、CUDA 或容器的 GPU
挂载问题。

## 4. 理解默认实验配置

顶层配置为：

```text
configs/config.yaml
```

它组合以下配置：

```text
configs/data/prototype.yaml
configs/model/flan_t5_base.yaml
configs/method/fedtaskprompt.yaml
configs/experiment/main.yaml
```

默认核心设置包括：

- backbone：`google/flan-t5-base`；
- Prompt 长度：20；
- 共享 basis 数量：8；
- 语义分组：任务向量余弦相似度，阈值 0.65；
- 本地适应步数：5；
- 元梯度：`coordinate_second_order`；
- 全局坐标生成器：唯一实例；
- basis 维护周期：20 轮；
- 主实验轮数：50。

命令行中的 Hydra 覆盖只改变本次实验，不会修改 YAML 文件。例如：

```bash
uv run python scripts/train.py experiment.num_rounds=2 method.inner_loop.steps=1
```

## 5. 运行代码测试

### 5.1 运行全部测试

```bash
uv run pytest -q
```

测试应覆盖：

- Prompt 合成与分解；
- basis 正交化与本地残差投影；
- 三种元梯度模式；
- 余弦任务语义聚类；
- 多任务簇共同更新唯一坐标生成器；
- 两轮端到端 smoke training。

### 5.2 分阶段排错

如果全部测试失败较多，可按依赖顺序运行：

```bash
uv run pytest -q tests/test_projection.py
uv run pytest -q tests/test_prompt_composition.py
uv run pytest -q tests/test_meta_gradients.py
uv run pytest -q tests/test_grouping.py
uv run pytest -q tests/test_smoke_train.py
```

所有测试通过后，再加载真实 Hugging Face 模型和数据。

## 6. 准备数据

默认原型包含情感分类、主题分类、实体抽取、问答和摘要任务。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_data.py data=prototype seed=42
```

该命令会下载或读取缓存数据，完成统一 text-to-text 预处理、客户端划分以及
support/query/test 划分，并生成：

```text
outputs/main/seed_42/data_manifest.json
```

检查 manifest 中以下字段：

- `num_tasks` 是否等于预期任务数；
- `num_clients` 是否等于任务数乘以每个数据集的客户端数；
- 每个客户端的 support、query 和 test 是否均非空；
- 不同实验是否使用相同 seed 和配置。

### 6.1 离线运行

确保模型和数据已在缓存中，然后使用：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_data.py \
  data.offline=true \
  model.local_files_only=true
```

如果缓存不完整，该命令会失败，而不会自动从网络补齐。

## 7. 先运行最小真实模型实验

不要直接启动 50 轮主实验。先使用小数据、少轮数和一步本地适应验证完整链路：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  experiment=main \
  seed=42 \
  device=cuda:0 \
  data.max_train_examples_per_dataset=60 \
  data.max_validation_examples_per_dataset=20 \
  data.max_test_examples_per_dataset=20 \
  data.partition.min_examples_per_client=12 \
  experiment.num_rounds=2 \
  experiment.eval_every_rounds=1 \
  experiment.eval_inner_steps='[0,1]' \
  method.inner_loop.steps=1 \
  checkpoint.save_every_rounds=1
```

完成后检查：

```text
outputs/main/seed_42/
├── resolved_config.yaml
├── rounds.jsonl
├── evaluations.jsonl
└── checkpoints/
    ├── round_000001.pt
    ├── round_000002.pt
    └── final.pt
```

重点确认：

1. `rounds.jsonl` 中轮数连续；
2. `active_groups` 非空；
3. `mean_support_loss` 和 `mean_query_loss` 为有限值；
4. `gradient_norm` 不是持续为零或 NaN；
5. `evaluations.jsonl` 同时包含 0 步和 1 步适应结果；
6. 显存不会随客户端数量持续线性增长。

## 8. 运行主实验

完成最小试跑后，运行默认主实验：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  experiment=main \
  seed=42 \
  device=cuda:0
```

为了提高可复现性，正式实验建议启用：

```bash
deterministic=true
```

完整命令为：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  experiment=main \
  seed=42 \
  device=cuda:0 \
  deterministic=true
```

确定性计算可能降低速度，并且部分算子只能给出警告。应在实验记录中注明是否启用。

## 9. 从 checkpoint 恢复训练

恢复训练时，`experiment.num_rounds` 表示最终总轮数，而不是额外增加的轮数。

例如，从第 20 轮恢复并训练到第 50 轮：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  experiment=main \
  seed=42 \
  device=cuda:0 \
  experiment.num_rounds=50 \
  checkpoint.resume_from=/absolute/path/to/round_000020.pt
```

恢复时必须保持以下项目一致：

- 数据配置与 seed；
- 客户端 ID 集合；
- 任务描述；
- 语义聚类配置；
- Prompt 长度和 basis 数量；
- 模型结构。

不要用不同数据划分或不同任务集合强行加载旧 checkpoint。

## 10. 独立评测 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py \
  experiment=main \
  seed=42 \
  device=cuda:0 \
  checkpoint.resume_from=/absolute/path/to/final.pt
```

评测步数由 `configs/experiment/main.yaml` 中的 `eval_inner_steps` 控制，也可以覆盖：

```bash
experiment.eval_inner_steps='[0,1,3,5,10]'
```

输出文件为：

```text
outputs/main/seed_42/standalone_evaluation.json
```

当前 evaluator 输出：

- 加权平均 test loss；
- 客户端 test loss 标准差；
- 最差客户端分位 test loss；
- 每个客户端的 test loss 和残差能量。

## 11. 运行冷启动实验

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_cold_start.py \
  experiment=cold_start \
  seed=42 \
  device=cuda:0
```

当前协议为：

- 每个任务保留一个客户端；
- 保留客户端的任务描述可见；
- 保留客户端的数据和训练更新不参与服务器训练；
- 训练结束后在 0、1、3、5、10 步适应下评测保留客户端。

输出：

```text
outputs/cold_start/seed_42/cold_start.json
```

该实现属于任务描述可见的传导式语义冷启动。论文中应明确说明，不能将它描述为任务描述也不可见的
完全归纳式冷启动。

## 12. 构造经验迁移矩阵

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_transfer_matrix.py \
  experiment=main \
  seed=42 \
  device=cuda:0
```

输出：

```text
outputs/main/seed_42/transfer_matrix.json
```

矩阵元素定义为：

```text
目标客户端本地 Prompt 的 test loss
-
源客户端 Prompt 在目标客户端上的 test loss
```

因此，正值表示源客户端 Prompt 相比目标客户端本地 Prompt 获得更低 loss，负值表示负迁移。
脚本同时输出客户端任务编码的余弦相似度矩阵，可用于分析语义相似度预测正迁移的能力。

该实验复杂度约为 $O(N^2)$ 次客户端评测。客户端较多时应先减小任务数量或客户端数量，
确认链路后再运行完整矩阵。

## 13. 运行当前支持的消融实验

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_ablation.py \
  experiment=ablation \
  seed=42 \
  device=cuda:0
```

当前能够严格运行的消融包括：

| 消融名称 | 实际覆盖配置 |
| --- | --- |
| `full` | 完整方法 |
| `single_group` | 将语义分组阈值设为 -1 |
| `no_local_residual` | 本地残差学习率与衰减设为 0 |
| `no_orthogonality` | 关闭每步本地残差投影 |
| `first_order` | 使用一阶元梯度 |
| `full_second_order` | 使用完整二阶元梯度 |
| `fixed_basis` | 关闭 basis 维护 |
| `no_center_update` | basis 维护时不更新中心 |

以下配置目前会被明确跳过：

- `random_group`；
- `zero_coordinate_init`；
- `group_mean_init`；
- `private_only`；
- `no_direction_replacement`。

跳过这些项目是为了避免用不等价的配置伪装成论文消融。需要得到完整消融表时，应先在算法层增加对应开关。

## 14. 使用两到三张 GPU 运行多个随机种子

正式结果至少建议运行 3 个随机种子。例如使用三张 GPU：

```bash
uv run python scripts/launch_sweep.py \
  --gpus 0 1 2 \
  --seeds 1 2 3 \
  --experiment main
```

使用两张 GPU 运行五个种子：

```bash
uv run python scripts/launch_sweep.py \
  --gpus 0 1 \
  --seeds 1 2 3 4 5 \
  --experiment main
```

调度器会在每个进程中设置 `CUDA_VISIBLE_DEVICES`，并让该进程使用逻辑设备 `cuda:0`。
一张 GPU 同一时间只运行一个实验。任务完成后，该 GPU 会继续处理队列中的下一个 seed。

建议先单独执行一次 `prepare_data.py`，避免多个进程同时下载同一数据集或模型。

## 15. 结果整理原则

### 15.1 随机种子

同一方法和对比方法必须使用相同 seed 集合。当前 seed 同时影响：

- 数据划分；
- 客户端采样；
- Prompt 和生成器初始化；
- 本地 batch 顺序。

因此，比较两个方法时不能随意使用不同 seed。

### 15.2 当前脚本可直接整理的内容

在当前 loss 级别实验中，可以直接从现有输出整理：

- 平均 test loss；
- 客户端间标准差；
- 最差 10% 客户端 loss；
- 0 步与若干适应步数的性能；
- 平均 support/query loss；
- 生成器梯度范数；
- 残差能量；
- basis 维护事件与触发次数。

以下内容虽然已有独立统计组件，但尚未接入主训练日志：

- 训练 wall-clock；
- 峰值显存；
- 理论通信量；
- 相对 Local Prompt 的增益与负迁移率。

这些指标必须先接入 simulator 和统一基线 runner，再用于论文正式效率表与迁移表。

正式论文表格应报告多个随机种子的均值和标准差，而不是只报告最好的一次运行。

### 15.3 任务原始指标

`metrics/task_metrics.py` 已提供 Accuracy、Exact Match、Token F1、Span F1 和 ROUGE-L，
但当前 evaluator 尚未调用模型生成并计算这些指标。接通生成评测前，论文中不要声称当前脚本已经输出这些任务指标。

## 16. 可复现性检查清单

每组正式实验应保存：

- `pyproject.toml`；
- `uv.lock`；
- `resolved_config.yaml`；
- Git commit ID；
- Python、PyTorch、CUDA、Transformers 和 Datasets 版本；
- GPU 型号；
- seed；
- 模型 revision；
- 数据集 revision；
- checkpoint；
- 原始 JSONL 日志；
- 汇总脚本和最终表格。

当前模型 revision 默认为 `main`。正式复现实验最好将其替换为固定 commit，而不是长期依赖会变化的
分支名称。

## 17. 常见问题

### 17.1 显存不足

按以下顺序降低开销：

1. 降低 `method.inner_loop.support_batch_size`；
2. 降低 `method.inner_loop.query_batch_size`；
3. 缩短 `data.text_template.max_source_length`；
4. 缩短 `data.text_template.max_target_length`；
5. 减少 `method.inner_loop.steps`；
6. 将元梯度切换为 `first_order`；
7. 使用更小的 seq2seq backbone。

示例：

```bash
method.inner_loop.support_batch_size=2 \
method.inner_loop.query_batch_size=2 \
data.text_template.max_source_length=256 \
method.inner_loop.meta_gradient=first_order
```

### 17.2 GPU 不支持 BF16

覆盖模型 dtype：

```bash
model.dtype=float32
```

顶层 `precision` 字段当前主要用于实验记录，实际 backbone dtype 由 `model.dtype` 控制。

### 17.3 模型或数据下载失败

检查网络、缓存目录和 Hugging Face 权限。对于需要授权的数据或模型，应先在运行环境中完成授权，
再启动 sweep。

### 17.4 恢复后分组不一致

说明当前客户端集合、任务描述、seed 或分组配置与 checkpoint 不一致。应恢复原配置，不能绕过一致性检查。

### 17.5 完整二阶模式速度很慢或显存很高

这是预期现象。`full_second_order` 只应用于小规模正确性与代价对照。主实验优先使用
`coordinate_second_order`，资源紧张时使用 `first_order`。

### 17.6 为什么没有真实网络通信

本项目只模拟 server/client 算法边界，并根据跨边界张量统计理论通信量。它不启动 RPC、NCCL 或真实
联邦节点，这与论文原型验证目标一致。

## 18. 推荐的完整执行顺序

```text
1. uv lock && uv sync --group dev
2. uv run pytest -q
3. prepare_data.py
4. 两轮小数据真实模型试跑
5. 单 seed 主实验
6. checkpoint 独立评测
7. 冷启动实验
8. 迁移矩阵实验
9. 当前支持的消融实验
10. 两到三张 GPU 并行运行多个 seed
11. 汇总均值、标准差、尾部性能、负迁移和效率
12. 接通统一基线 runner 与生成指标后，再制作论文最终对比表
```

只有在前一步的输出、数值范围和资源占用均正常后，才应扩大数据规模、训练轮数和随机种子数量。
