# FedTaskPrompt

FedTaskPrompt 是论文《以任务语义作为协作先验：面向异构联邦软提示学习的仿射共享与正交个性化》的实验验证原型。

本项目关注算法假设和实验结论，不模拟真实跨机构网络。所有客户端在单个 Python 进程中顺序执行，并复用同一个冻结的 Hugging Face backbone。服务器、客户端、数据和算法状态在代码层面保持独立，但不会为每个客户端复制一份语言模型，也不依赖 RPC、NCCL 或真实分布式部署。

## 1. 研究目标

框架用于验证以下问题：

1. 训练前的任务语义能否为新客户端提供合理的初始分组；
2. 组内低维仿射 Prompt 子空间能否促进相关任务之间的迁移；
3. 与共享方向正交的本地残差能否降低共享分量与个性化分量之间的干扰；
4. 支持集适应和查询集反馈能否修正语义生成的初始坐标；
5. 仅保留共享坐标路径的低维二阶梯度是否具有合理的性能—开销折中；
6. 周期性 basis 维护能否吸收多个客户端反复出现的本地变化；
7. FedTaskPrompt 能否在提高平均性能的同时降低客户端负迁移率。

这不是生产级联邦学习系统。隐私、安全聚合、容错、网络传输和真实客户端并发不属于第一阶段目标。

## 2. 原型的核心约束

### 2.1 单模型实例

所有客户端共享一个冻结的 `transformers` 模型实例：

```text
server selects client
        |
        v
load group prompt state + client residual
        |
        v
run support adaptation and query evaluation
        |
        v
return coordinate feedback; release temporary graph
```

客户端对象只保存轻量状态，例如任务描述、数据索引、所属组、持久化评测残差和统计量。训练轮次中的本地残差默认从零开始，完成查询反馈或 basis 维护后即释放。

### 2.2 单进程联邦模拟

一个训练轮次包含以下步骤：

1. 服务器采样参与客户端；
2. 服务器根据任务语义为客户端分组，并生成初始坐标；
3. 客户端在支持集上执行 $H$ 步局部适应；
4. 客户端在查询集上计算查询损失和坐标反馈；
5. 服务器在每个组内聚合反馈并更新坐标生成器；
6. 满足维护条件时，服务器更新组中心或共享方向；
7. 定期在客户端测试集上评价性能、负迁移和效率。

该流程保留了论文的 server/client 边界，但函数调用均发生在本地进程中。

### 2.3 GPU 使用方式

默认配置只需要一张 GPU。两到三张 GPU 用于并行运行不同随机种子、数据设置或消融实验，而不是把一个联邦训练任务拆成真实多机通信。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py experiment=main seed=1
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py experiment=main seed=2
CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py experiment=main seed=3
```

单个实验内部默认不使用 DDP。后续如需同时调度多组实验，可由 `scripts/launch_sweep.py` 为每张 GPU 启动一个独立子进程。

## 3. Prompt 建模

对组 $r$，服务器维护基础 Prompt $B_{0,r}\in\mathbb{R}^{L_p\times d}$ 和 $K$ 个共享方向：

\[
Q_r=[\operatorname{vec}(B_{r,1}),\ldots,\operatorname{vec}(B_{r,K})]
\in\mathbb{R}^{D\times K},\qquad D=L_p d.
\]

客户端 $n$ 的 Prompt 为

\[
P_n=B_{0,a_n}+\sum_{k=1}^{K}c_{n,k}B_{a_n,k}+U_n,
\qquad Q_{a_n}^{\top}\operatorname{vec}(U_n)=0.
\]

任务编码器将结构化任务描述映射为 $z_n$，组原型映射为 $\mu_r$。坐标生成器输出

\[
c_n^{(0)}=G_{\theta_r}(z_n,\mu_r).
\]

在支持集上，客户端联合更新 $c_n$ 和 $U_n$，并在每次残差更新后投影到共享方向的正交补。查询集只向服务器提供关于初始坐标的反馈。

项目会同时实现三种元梯度模式：

- `first_order`：忽略内循环 Hessian；
- `coordinate_second_order`：只保留共享坐标路径上的 $H_{cc}v$，对应论文的低维二阶近似；
- `full_second_order`：保留联合变量 $(c,U)$ 的完整计算图，仅用于小规模正确性和代价对照。

其中 `coordinate_second_order` 会显式截断本地残差相关的交叉二阶路径，避免把查询端的 `stop_gradient(U)` 错误解释为自动消除了整个内循环中的交叉依赖。

## 4. 项目结构

```text
fedtaskprompt/
├── README.md
├── pyproject.toml
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   └── prototype.yaml
│   ├── experiment/
│   │   ├── main.yaml
│   │   ├── cold_start.yaml
│   │   ├── ablation.yaml
│   │   └── basis_drift.yaml
│   ├── model/
│   │   └── flan_t5_base.yaml
│   └── method/
│       └── fedtaskprompt.yaml
├── data/
│   ├── __init__.py
│   ├── schema.py
│   ├── registry.py
│   ├── preprocess.py
│   ├── partition.py
│   └── federated_data.py
├── model/
│   ├── __init__.py
│   ├── backbone.py
│   ├── soft_prompt.py
│   ├── task_encoder.py
│   ├── coordinate_generator.py
│   └── prompt_subspace.py
├── client/
│   ├── __init__.py
│   ├── state.py
│   ├── inner_loop.py
│   └── client.py
├── server/
│   ├── __init__.py
│   ├── grouping.py
│   ├── group_state.py
│   ├── aggregation.py
│   ├── basis_maintenance.py
│   └── server.py
├── trainer/
│   ├── __init__.py
│   ├── simulator.py
│   ├── evaluator.py
│   └── checkpoint.py
├── metrics/
│   ├── __init__.py
│   ├── task_metrics.py
│   ├── transfer.py
│   ├── negative_transfer.py
│   └── efficiency.py
├── baselines/
│   ├── __init__.py
│   ├── local_prompt.py
│   ├── fedavg_prompt.py
│   ├── shared_local_prompt.py
│   ├── ifca_prompt.py
│   └── per_fedavg_prompt.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── build_transfer_matrix.py
│   ├── run_cold_start.py
│   ├── run_ablation.py
│   └── launch_sweep.py
├── tests/
│   ├── test_projection.py
│   ├── test_prompt_composition.py
│   ├── test_meta_gradients.py
│   ├── test_grouping.py
│   └── test_smoke_train.py
└── outputs/
    └── .gitkeep
```

## 5. 模块职责

### `data/`

- 定义统一的 text-to-text 样本格式；
- 注册不同任务数据集及其评价指标；
- 构造 support/query/test 划分；
- 将一个数据集划分为多个模拟机构客户端；
- 保存结构化任务记录 `op/in/out/dom/lang`；
- 保证客户端只通过自己的 dataloader 访问本地样本。

统一样本至少包含：

```python
{
    "input_text": str,
    "target_text": str,
    "task_id": str,
    "client_id": str,
    "metadata": dict,
}
```

### `model/`

- `backbone.py`：加载并冻结共享语言模型；
- `soft_prompt.py`：将连续 Prompt 注入输入 embedding；
- `task_encoder.py`：编码客户端任务描述与组原型；
- `coordinate_generator.py`：预测客户端初始坐标；
- `prompt_subspace.py`：Prompt 合成、方向正交化和残差投影。

`SharedBackbone` 在整个训练进程中只实例化一次。任何客户端类都不能持有独立 backbone 副本。

### `client/`

- 保存轻量客户端元信息；
- 执行 support-set 内循环；
- 计算 query loss；
- 根据配置产生一阶、坐标二阶或完整二阶反馈；
- 统计本地残差能量和共享坐标使用程度。

客户端计算结束后必须释放临时优化器、计算图和本轮残差，以控制显存峰值。

### `server/`

- 使用结构化任务语义完成初始分组；
- 管理每个组的 $B_0,Q,G_\theta$；
- 在组内聚合查询反馈；
- 创建新组和处理冷启动客户端；
- 根据残差能量触发低频 basis 维护。

### `trainer/`

- 编排联邦轮次，但不包含具体算法公式；
- 负责客户端采样、评测、日志和 checkpoint；
- 支持从 checkpoint 恢复；
- 保证相同客户端采样序列可被不同基线复用。

### `baselines/`

所有基线复用相同的 backbone、数据划分、Prompt 长度、客户端采样序列和评价器。第一阶段实现以下最低基线集合：

1. `LocalPrompt`；
2. `FedAvgPrompt`；
3. `SharedLocalPrompt`；
4. `IFCAPrompt`；
5. `PerFedAvgPrompt`；
6. `FedTaskPrompt`。

### `metrics/`

除任务原始指标外，统一计算：

- 相对 Local Prompt 的客户端平均增益；
- 负迁移率；
- 最差 10% 客户端性能；
- 客户端间标准差；
- 语义相似度预测正迁移的 AUROC；
- 理论通信字节数；
- wall-clock time 和峰值显存。

## 6. 配置系统

项目使用 Hydra 管理实验配置。顶层配置由数据、模型、方法和实验四部分组成：

```yaml
defaults:
  - data: prototype
  - model: flan_t5_base
  - method: fedtaskprompt
  - experiment: main
  - _self_

seed: 42
device: cuda
output_dir: outputs/${experiment.name}/seed_${seed}
```

重要算法参数包括：

```yaml
method:
  prompt_length: 20
  num_basis: 8
  inner_steps: 5
  inner_lr_coordinate: 0.1
  inner_lr_residual: 0.01
  server_lr: 0.001
  meta_gradient: coordinate_second_order
  orthogonalize_basis: true
  project_local_residual: true
  reset_residual_each_round: true
  basis_maintenance_interval: 20
  residual_energy_threshold: 1.0
  basis_energy_ratio: 0.9
  max_replaced_basis: 2
```

配置文件必须记录所有影响实验可复现性的参数。命令行只用于覆盖配置，不在训练脚本中硬编码超参数。

## 7. 数据配置

第一阶段先支持小规模、可快速完成的验证设置：

- 情感分类；
- 主题分类；
- 实体抽取；
- 问答；
- 摘要生成。

每类任务选择一到两个数据集，每个数据集模拟 2–3 个机构客户端。为了先验证训练链路，`prototype.yaml` 默认只抽取少量样本；完整实验再通过配置扩大样本数。

每个客户端的数据严格划分为：

```text
support: 本地适应
query:   元训练反馈
test:    最终评价
```

测试集不得参与本地更新或服务器参数选择。

## 8. 训练状态与生命周期

### 服务器持久化状态

- 任务组和组原型；
- 每组基础 Prompt；
- 每组共享 Prompt 方向；
- 每组坐标生成器及其优化器；
- basis 使用统计与维护计数器；
- 全局轮次和随机数状态。

### 客户端持久化状态

- 客户端 ID；
- 任务记录；
- 本地数据划分；
- 当前组 ID；
- 可选的部署阶段个性化残差；
- 历史评价指标。

### 每轮临时状态

- $c_n^{(0)},\ldots,c_n^{(H)}$；
- $U_n^{(0)},\ldots,U_n^{(H)}$；
- support/query 计算图；
- Hessian-vector product 中间量。

临时状态不能写入常规 checkpoint，避免 checkpoint 随客户端数量线性膨胀。

## 9. 通信量的模拟

虽然原型不会实际发送张量，但会根据真正跨 server/client 边界的对象计算理论通信量。

常规轮次分别记录：

- 服务器向客户端发送的初始坐标；
- 客户端上传的坐标反馈；
- 可选的终端坐标统计。

basis 维护轮额外记录：

- 裁剪后的本地残差；
- 更新后的基础 Prompt 或共享方向。

报告中同时给出：

- 常规轮次通信量；
- 维护轮通信量；
- 按整个训练过程摊销后的平均通信量。

## 10. 实验入口

安装依赖：

```bash
uv sync
```

准备原型数据：

```bash
uv run python scripts/prepare_data.py data=prototype
```

运行主实验：

```bash
uv run python scripts/train.py experiment=main
```

运行冷启动实验：

```bash
uv run python scripts/run_cold_start.py experiment=cold_start
```

运行消融：

```bash
uv run python scripts/run_ablation.py experiment=ablation
```

构造经验迁移矩阵：

```bash
uv run python scripts/build_transfer_matrix.py
```

以上命令将在对应脚本实现后生效。

## 11. 最小验证顺序

为了避免直接在完整 NLP 数据集上排查算法错误，推荐按以下顺序开发：

1. 实现 Prompt 合成和正交投影，并通过解析单元测试；
2. 使用极小的合成 text-to-text 数据验证单客户端 support/query 更新；
3. 用有限差分或完整 autograd 对照三种元梯度；
4. 运行两个组、四个客户端的 smoke experiment；
5. 加入 Local Prompt 和 FedAvg-Prompt 基线；
6. 接入真实 Hugging Face 数据集；
7. 增加冷启动、负迁移和 basis 维护实验；
8. 最后扩大模型、客户端和随机种子规模。

## 12. 实现原则

- 算法公式放在 `model/`、`client/` 和 `server/`，训练编排放在 `trainer/`；
- 数据集特有逻辑不能进入客户端或服务器类；
- 客户端对象不得复制 backbone；
- 所有跨边界张量都经过通信统计器登记；
- 所有实验使用相同的数据划分和客户端采样计划；
- 每个关键数学操作必须有独立测试；
- 先保证小规模结果正确，再进行混合精度和性能优化；
- 默认不声称提供形式化隐私保护。

## 13. 后续文件输出顺序

后续将按依赖关系依次创建文件，而不是一次性堆叠全部实现：

1. `pyproject.toml` 与基础配置；
2. `data/schema.py` 和数据注册接口；
3. `model/prompt_subspace.py` 与对应测试；
4. `model/backbone.py`、`soft_prompt.py`；
5. `model/task_encoder.py`、`coordinate_generator.py`；
6. `client/state.py`、`inner_loop.py`、`client.py`；
7. `server/group_state.py`、`grouping.py`、`aggregation.py`；
8. `server/basis_maintenance.py`、`server.py`；
9. `trainer/simulator.py`、评价与 checkpoint；
10. 基线、运行脚本和端到端 smoke test。

每一阶段都会先说明文件职责，再给出代码和最小验证方式。
