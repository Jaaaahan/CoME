# CoME 离线观察与 GraphMemix

CoME 只负责从原始来源提取、检查和保存与问题无关的观察。GraphMemix 继续负责向量召回、记忆图、节点验证、ECV、森林选择、剪枝和回答。实现依据是 [集成方案](come_graphmemix_integration_plan.md)。

默认 `configs/release/graphmemix.json` 开启观察扩展；`graphmemix_legacy.json` 保留基线参数。两者的方法预算参数相同，Judge 协议分别为 1.3 和 1.2。2026-10-26 的修改限定为观察的区域图文编码，以及将 memory/observation 都作为图节点；继续采用原来的离线 mutual-kNN 和在线一跳邻接扩展。旧版完成过小样本 GPU 链路验证；本次修改通过 CPU 回归与真实 FAISS 链路验证，尚未重跑 GPU 或完整 benchmark。原有发布结果不代表新版效果。

## 模块边界

| 模块 | 职责 |
|---|---|
| `come/types.py:PreprocessConfig` | 只包含观察条数、图像大小、文本分块和 grounding 裁剪设置。 |
| `come/models.py:ObservationExtractor` | `observe(image, store)` 做离线提取和来源 grounding；没有问题、gap、查询 state 入参，也没有问答接口。 |
| `come/source_media.py`、`come/store.py` | 原始来源、文本区间、EXIF 方向、模型图像准备、裁剪和内部观察写入。 |
| `come/pipeline.py:ObservationPreprocessor` | 按图片续跑纯预处理，返回每条原始记忆的观察字段。 |
| `come/runner.py` | 只读取 memories/assets，不读取 questions，不加载 GME 或 reader。 |
| `come/fields.py:ObservationSnapshot` | 验证已完成快照，按 `context_id + memory_id` 挂载字段；接口只读。 |
| `observation_evidence.py` | GraphMemix 的有界展示规则、来源文本展示及展示协议。 |
| `methods/concrete_memix.py` | 追加观察向量表示、记忆级最高分合并、Reader 文本。 |
| `retrieval_units.py`、`methods/gme.py` | 稳定 unit 身份、观察原图定位、EXIF 区域裁剪及图文联合编码。 |
| `unit_graph.py`、`unit_evidence.py` | 重建 memory/observation 图向量，将实际命中 unit 接入原有候选算法，独立传递到评分、选择和 Reader。 |
| `memory_scope.py` | GraphMemix 的 all/prefix/ids 可见范围。控制信息不进入问题文本。 |

旧的查询实验保存在 `come/legacy_models.py`、`legacy_pipeline.py`、`legacy_runner.py`，以及旧 `retrieval.py`、`selection.py`、`packet.py` 中，用于历史回归。它们不在发布入口调用链中。`CoMEConfig` 是历史查询配置，不能传给新预处理入口。共享的 `evidence_forest.py`、`evidence_verifier.py` 继续供 GraphMemix 使用。

## 运行

```bash
export PYTHONPATH=src:scripts

# 检查配置和完整阶段编排，不调用模型。
python scripts/run_graphmemix_release.py validate-config
python scripts/run_graphmemix_release.py benchmark \
  --dataset atm --backbone qwen3vl8b --to-stage reader --dry-run

# 原有 GraphMemix 基线。
python scripts/run_graphmemix_release.py \
  --config configs/release/graphmemix_legacy.json benchmark \
  --dataset atm --backbone qwen3vl8b --to-stage reader --dry-run
```

阶段顺序：

```text
observations（开关开启时执行）
→ checkpoint → priors → relations → node → ecv → select → prune → reader → judge
```

`--from-stage` 默认为 `observations`。从 `checkpoint` 或之后启动增强模式时，必须已有完整、兼容的观察快照；不会在后台另行提取。旧 `pipeline=come-v1` 配置会明确报错，旧集群实验启动脚本和缓存不能直接用于新流程。

独立预处理也可运行，无需问题文件：

```bash
python -m mm_memory_bench.come.runner \
  --config configs/release/graphmemix.json \
  --bundle /path/to/bundle \
  --artifact-root /path/to/extraction \
  --output /path/to/observations.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1
```

`observations` 配置控制启用状态、提取模型和生成预算；`model` 显式指定独立提取模型，`base_url`、`reasoning_effort` 可覆盖提取服务设置。默认提取模型为 Qwen3-VL-8B；使用其他 Reader 时可先完成 observations，或通过 `observations.base_url` 指定独立提取服务。`come` 只接收 `PreprocessConfig` 字段。提取契约独立记录模型、提示词协议和预处理设置；修改问题集、embedding 或 Reader 不会使已完成的观察失效。

默认增强产物分别位于：

```text
artifacts/graphmemix/observations-v1/
  extraction/{dataset}/contexts/{context_hash}/
  observations/{dataset}.jsonl
  observations/{dataset}.manifest.json
  checkpoints/{dataset}/
  source_priors/{dataset}.jsonl
  relation_store/{dataset}.jsonl
runs/release/graphmemix/observations-v1/{backbone}/{dataset}/
```

离线观察与 checkpoint 不按 Reader backbone 分目录，同一数据和提取设置可跨 Reader 复用。caption sidecar 和外部 Memix 源码仍从原有路径读取。基线和历史 `come-v1` 产物保留原路径。

## 快照与来源

每条 JSONL 记录包含 `context_id`、`memory_id` 和 `come`，后者包含 `schema_version=1`、`producer_protocol=come-observation-field-1`、`source_digest`、`observations`。

每个观察保存 ID、版本、描述、事实、原图摘要、原图尺寸、像素区域和独立可解析的来源引用。文本引用包含原始内容块编号、字符起止位置、完整原文的版本摘要及精确原文片段。图像引用包含内容块编号、资产 ID 和源文件摘要。引用不依赖旧 CoME 的候选节点 ID。

模型只输出 `normalized_1000` 坐标，提取时转换一次。外部字段明确标记 `oriented_pixel_xyxy`，所有后续操作使用 EXIF 方向处理后的原图像素。旧报告中缺少明确坐标协议的值不会被猜测转换。

首版仅接受本条记忆内的来源引用。跨记忆引用明确拒绝；未显示必要原文的身份、归属观察会被省略。同来源、同事实的重复值去重，冲突事实值分别保留。

完成 manifest 绑定整个 sidecar 摘要及来源摘要，最后写入。每条记忆都必须有记录，包括成功处理后的空观察列表。空列表正常回退到原有证据；缺行、提取失败、缺失 manifest、来源变更、坏引用、不兼容版本均报错，不能假装增强流程已完成。

## GraphMemix 接入

`_resolved_memory(..., observations=snapshot)` 在安全筛选后挂载已校验字段，并通过 `observation_source_images` 保存按原始内容位置解析的图像。即使 `derived` 视图移除了原图，观察仍能读取自己的原始来源，不能把改变后的内容位置当成来源位置。

`_adapt_memory()` 保留原有记忆表示，为每条观察增加一个 `representation=come_observation_multimodal` unit：文本为 `description`，图像为来源图在 EXIF 纠正后的区域裁剪。多个区域使用联合包围框，保留区域之间的空间关系；一个观察始终对应一个向量。GME 调用 `get_fused_embeddings(texts, images)`，不再使用观察的纯文本 embedding。

unit 身份是 JSON 编码的 `[memory_id, observation_id]`；完整记忆的第二项为 `null`。完整 memory 的多张图/视频帧是同一个逻辑 unit 的多个检索视图，构图向量使用原来的归一化均值；atomic text 仍属于该 memory。观察有独立向量，不混入父 memory 的均值。

检索使用同一个 FAISS 索引，保留有序 `unit_hits`（unit ID、memory ID、类型、观察 ID/版本、分数）。原有 memory 级 source priors 仍按最高命中分数聚合并自适应预取，供来源级召回统计使用；图种子与后续评分直接使用 `unit_hits`，不会先将同一 memory 的观察压缩成一个根，也不会把父 memory 的最高分复制给所有观察。`observation_hits` 继续作为兼容诊断字段。

`relations` 阶段继续对每个 context 的所有图节点执行原来的 `mutual_neighbors()`：排除自身，计算 top-k，仅在双方互为邻居时保存语义边及双方排名。变化仅是输入节点从 memory 扩展到 memory + observation。节点类型不改变排名规则，同属 memory 的观察也正常参与近邻排名。

同轮次、相邻对话和可选时空关系仍连接完整 memory 节点；每个 observation 与父 memory 增加 `belongs_to` 显式关系，沿用原来的显式边权重。关系文件保存每个 context 的 unit 元数据、离线边和 checkpoint 摘要；ECV/森林读取 unit 边，原始 memory 显式关系记录仍保留供旧流程使用。

node 阶段将 unit ID 直接传给原来的 `candidate_pool()`，不另写扩图算法：

1. 根据召回分数取前 `source_top_l` 个实际命中的 unit，默认 24 个；同一 memory 的多个观察可以同时成为种子。
2. 沿这些种子已经存储的边扩展一跳。语义边仍按双方排名过滤，默认 `semantic_k=8`；查询阶段不加载图向量索引、不重新搜索近邻。
3. 按原有边相似度、种子排名和节点 ID 排序，对相同 unit ID 去重并补充至 `candidate_limit`，默认 48 个 unit。可见范围过滤、图模式和一跳限制沿用原逻辑；过滤后不重新计算邻居，不增加目标 memory 配额。
4. 候选 unit 直接传给节点验证、ECV、森林选择和剪枝，再将选中的 unit 交给 Reader。同一 memory 的多个观察、以及完整 memory 本身均保留各自的 ID、分数和内容。node 缓存记录 `seed_unit_ids`、`candidate_unit_ids` 和图关系；父 memory 只出现在溯源元数据时，不自动成为候选或新的扩展根。

因此，24/48 和最终选择的 k 都按 unit 计数。增强配置 `configs/release/graphmemix.json` 的 `reader_k` 默认为 **20**，即 24 个种子 → 最多 48 个候选 → 最多 20 个最终 unit；冻结的 legacy 配置仍为 10。节点验证、ECV、森林和 Reader 的评分公式保持不变。精确剪枝固定保留独立净收益为正的节点，只枚举其余节点；没有有效边时直接求独立节点的最优非空集合，保持原目标和精确性，避免 20 个节点都做全子集枚举。剪枝报告按实际输入 Top-K 计算召回，不再固定截断为 10。选择与剪枝输出的 `retrieved_unit_ids` 是后续唯一的证据输入，Reader 将它保存在预测的 `metadata.retrieved_unit_ids`。另存的 `retrieved_memory_ids` 只用于按原 benchmark 标注统计来源级召回，不参与下游选择或 agent 输入。关闭 observations 时继续使用原有 memory 图。

建库、priors、关系构建、节点验证、ECV、选择、剪枝和 Reader 的脚本都接受 `--observations`。节点验证与 ECV 独立加载 sidecar；Reader 校验 checkpoint 的观察契约后重新挂载同一快照，支持正常多模态、text-only 和 action-aware 三种模式。

## 展示预算与缓存

- verifier/ECV 的 `snippet` 仍为默认 420 字符，观察最多占 `min(300, snippet预算×3/4)`。每个 observation 另外携带 `source_context`，来源上下文不挤占观察摘要预算；ECV 的候选和 anchor 都展示同一结构。
- `source_context` 从快照校验过的原始记忆生成，包含来源记录时间 `recorded_timestamp`、session/source/round 标识（有值时）、说话者及最多 600 字符的原始文本。超长原文明确标记 `original_text_truncated=true`。记录时间不自动等同于事件时间，原文片段也不表示完整身份依据；带来源引用的观察事实仍遵守原有完整保留或省略规则。
- Reader 使用同一份来源上下文、原有 1200 字符观察预算及来源区域图像。上下文独立于该观察预算，三种 Reader 模式都展示它。上下文从原始 bundle 恢复，避免把 checkpoint 的 derived caption 当作原文；不自动加入兄弟观察，也不把 observation 合并回父 memory。
- text-only 模式只发送文本，action-aware 模式继续按原有 text/low/high 动作控制图像输入。总上下文窗口仍为 32768 tokens，输出预算、OCR 预算和 token 检查保持不变。
- 展示文本明确标记为模型观察，不是真值。节点/ECV 的 `observation_ids` 和 Reader 的 `metadata.observation_ids` 按 unit ID 记录实际展示的观察；Reader 的 `query_added_observations` 为 0。

checkpoint 绑定观察契约、编码协议、embedding 身份和 memory view。当前编码协议为 `graphmemix-observation-region-description-2`，图协议为 `graphmemix-offline-unit-graph-2`。priors 绑定 state/index 摘要；unit 图校验观察契约及 prior 对应的 checkpoint 索引，防止混用不同向量空间。node/ECV 绑定关系文件摘要、观察快照和渲染协议，其中 `evidence_identity=graphmemix-unit-evidence-1` 区分独立 unit 与旧的父 memory 合并输入。选择、剪枝记录输入摘要，Reader 绑定 checkpoint、选择记录、问题文件和渲染协议；judge 已按预测行摘要判断缓存。

不兼容缓存会明确拒绝续跑，请使用新产物路径重建受影响阶段。更换 embedding 不要求重做观察；修改源文件或提取设置则必须先生成新的观察快照。查询阶段不会调用提取或 grounding，不写观察，也不更新索引。

来源上下文变更使用 `graphmemix-observation-evidence-2` 渲染协议，并把上下文版本、原文预算和截断规则写入契约。升级后可复用原观察、embedding checkpoint、priors 和离线图；使用新的 `--run-root`，从 `--from-stage node` 开始重新生成节点分数、ECV、选择、剪枝、Reader 和判分。不要复用缺少来源上下文的旧评分或旧答案。

从纯文本观察版本迁移时，可以复用原有 observations JSONL 和 manifest，但必须重建 checkpoint、priors、relations 及后续查询产物。如果已按上一版生成图文 checkpoint 和包含 `unit_hits` 的 priors，这两者可以复用，只需重建 relations 及之后的产物。之前“查询时计算邻居”的 `unit_index` 关系文件会被拒绝；新版关系文件不依赖额外的 `.units-*.faiss` 文件。

如果已经具备当前图文 checkpoint、`unit_hits` priors 和离线 unit 图，本次取消父 memory 合并只需从 node 阶段重新运行至 Reader，离线产物可以复用。旧的 memory 级选择结果不会被当成 unit 选择结果读取。

## 验证与效果对照

```bash
PYTHONPATH=src:scripts python -m pytest -q tests
python scripts/run_graphmemix_release.py validate-config
```

`tests/test_observation_integration.py` 覆盖无问题文件预处理、来源校验、非方形和 EXIF 图像、重复观察召回、多个问题向量下的预取、canonical vectors、预算、可见范围、checkpoint 恢复、实际 node/ECV/Reader 消息、缓存拒绝及只读边界。包含同一 memory 两个观察独立贯穿全部阶段的用例，并检查 Reader 三种模式中的独立内容、顺序、裁剪像素尺寸及跨机器资产路径重绑定。模型调用使用确定性测试替身，向量保存和检索使用真实 FAISS。

`tests/test_gme_observation_encoding.py` 验证真实 GME 适配器的图文调用、正确源图及 EXIF 裁剪；`tests/test_unit_graph.py` 验证离线全节点 mutual-kNN、双方排名过滤、查询阶段禁止近邻搜索、复用原有 top-L/一跳候选算法、多观察种子、可见范围、父子归属和缓存不兼容拒绝。

实际模型质量、Recall@200、候选 Recall@48、准确率和成本仍需在同一数据、模型、记忆视图、候选预算、上下文窗口和冻结系数下做 A/B 对照。观察描述中的错误会影响召回和证据判断，架构回归通过不代表效果提升。

## 本地 Slurm 实测

当前集群可用 `scripts/prepare_observation_gpu_smoke.py` 准备固定的小样本与源码副本，再提交 `scripts/observation_gpu_smoke.sbatch`。该脚本依赖本工作区的 `cluster/cuhk-suite/{token_proxy.py,check_server.py,process_lifecycle.sh}` 和已有本地模型、Conda 环境；它不是发布包的通用部署入口。

H2HMem 的三卡全量入口为 `scripts/h2hmem_3gpu.sbatch`：两卡 TP=2 运行
Qwen3-VL-8B，一卡运行 GME 编码，提取与查询并发均为 2。使用项目主评测的
1,982 题，保留全部 7,078 条记忆；准备时校验并复制已有的 1,300 张图片 caption。
以下命令创建独立的源码、输入和输出目录，默认申请 12 CPU、64G 内存、48 小时：

```bash
export COME_FULL_ROOT="$PWD/.runtime-h2hmem-3gpu-$(date +%Y%m%d-%H%M%S)"
PYTHONPATH=src:scripts /share/home/jyh/.conda/envs/graphmemix/bin/python \
  scripts/prepare_observation_full_run.py \
  --data-root /share/workspace3/yhjia/graphmemix --output "$COME_FULL_ROOT" \
  --datasets h2hmem --gpus 3 \
  --caption-sidecar /share/workspace3/yhjia/graphmemix/experiments/h2hmem-4090-20261017/artifacts/captions/h2hmem.jsonl
sbatch --export=ALL --output="$COME_FULL_ROOT/h2hmem/logs/slurm-%j.log" \
  "$COME_FULL_ROOT/source/scripts/h2hmem_3gpu.sbatch"
```

阶段和日志位于 `$COME_FULL_ROOT/h2hmem/phase.json`、`h2hmem/logs/`，
回答位于 `h2hmem/runs/observations-v1/qwen3vl8b/h2hmem/reader/predictions.jsonl`。
完成后自动检查离线产物未被查询修改，并在 `h2hmem/metrics/` 输出 H2H 原生
Precision、Recall、F1、BLEU-1 和本地 Qwen 判分；本地判分不等同于论文的 GPT-5-mini 判分。
旧作业结束后，可用同一 `COME_FULL_ROOT` 和冻结脚本重新提交续跑。

```bash
export COME_SMOKE_ROOT="$PWD/.runtime-observation-smoke"
PYTHONPATH=src:scripts /share/home/jyh/.conda/envs/graphmemix/bin/python \
  scripts/prepare_observation_gpu_smoke.py \
  --data-root /share/workspace3/yhjia/graphmemix --output "$COME_SMOKE_ROOT"
sbatch --export=ALL --output="$COME_SMOKE_ROOT/logs/slurm-%j.log" \
  "$COME_SMOKE_ROOT/source/scripts/observation_gpu_smoke.sbatch"

# 等作业完成后在 CPU 上审计，成功时生成 audit.json。
PYTHONPATH=src:scripts /share/home/jyh/.conda/envs/graphmemix/bin/python \
  scripts/audit_observation_gpu_smoke.py --run-root "$COME_SMOKE_ROOT"
```

样本保留 Mem-Gallery、MemEye 各 3 个原始问题的标注证据和部分干扰记忆，共 26 条记忆、6 张图。选样使用了 gold evidence，因此不能把命中率或回答表现当作正式 benchmark 分数。测试使用 standalone 后端和原有 `derived` 视图，执行 observations 到 Reader，不调用外部 judge。

当前审计检查观察与原始来源的契约、图文 unit 和裁剪区域、独立的父 memory 向量、图索引的观察覆盖、实际命中 unit 到图根的对应关系、unit 级预算与下游身份传递、各阶段覆盖率、真实 HTTP 消息中的观察，以及查询期间原始离线文件和索引逐字节不变。`structural_checks_passed` 仅表示这些检查通过，不表示每条模型观察语义正确。以下 GPU 结果来自 2026-10-19 的旧版纯文本观察实现。

2026-10-19 在 gpu03 的 3 张 RTX 4090 D 上完成两轮实测，作业 78875、78876 均正常退出，分别耗时 5 分 58 秒、5 分 42 秒。当前实现保留 39 条观察；试验性的更严格提取/grounding 提示词保留 22 条、拒绝 14 条。两轮都完成 6 个问题的全部阶段和结构审计，关闭服务后的预处理与 Reader 缓存续跑也通过。

人工抽查发现：模型将课堂图片上不清晰的黑板字样断言为 `CHEMISTRY`，模型 grounding 仍接受该观察。更严格提示词的复测仍未排除该误读，且 MemEye 的“最早广告品牌”问题出现节点评分、剪枝丢失正确证据的回归，答案从 Coca-Cola 变成 Burger King。因此未采用该提示词试改，当前代码保持 `come-offline-observe-ground-1`；试验源码与产物单独保存在 `.runtime-observations-gpu-20261019-v2/`。上述质量问题尚未解决。当前实现的实测产物位于 `.runtime-observations-gpu-20261019/`。
