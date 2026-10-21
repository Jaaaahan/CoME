# CoME 离线观察与 GraphMemix

CoME 只负责从原始来源提取、检查和保存与问题无关的观察。GraphMemix 继续负责向量召回、记忆图、节点验证、ECV、森林选择、剪枝和回答。实现依据是 [集成方案](come_graphmemix_integration_plan.md)。

默认 `configs/release/graphmemix.json` 开启观察扩展；`graphmemix_legacy.json` 保留基线参数。两者使用相同的 `benchmark_suite`。已完成小样本 GPU 链路验证，尚无完整 benchmark 效果评测；原有发布结果不代表增强模式的效果。

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

`_resolved_memory(..., observations=snapshot)` 在原有安全筛选后显式挂载已校验字段。`_adapt_memory()` 保持原有 `record.text` 和原有 embedding units，另加 `representation=come_observation_text` 的文本 units，记录观察 ID/版本。

检索使用同一个 FAISS 索引，按 `memory_id` 取最大相似度。观察不独占候选名额。自适应扩大预取，直到各问题视图中剩余行均无法超过当前前 k 条记忆，或索引耗尽。`observation_hits` 只作诊断，不参与第二套排序。

图仍是原始记忆图，使用原有种子和一跳邻居。所有带 `representation` 的附加 units 都被 canonical vector 构造排除。节点收益、ECV 端点、冻结校准参数、森林目标和剪枝规则保持记忆级语义。

建库、priors、关系构建、节点验证、ECV、选择、剪枝和 Reader 的脚本都接受 `--observations`。节点验证与 ECV 独立加载 sidecar；Reader 校验 checkpoint 的观察契约后重新挂载同一快照，支持正常多模态、text-only 和 action-aware 三种模式。

## 展示预算与缓存

- verifier/ECV 保持每条证据总计 420 字符的默认预算。观察最多占 `min(300, 总预算×3/4)`，先保留观察所需空间，再渲染原有证据。
- 每条记忆最多展示 4 个观察，按观察 ID 稳定排序。每个观察与其必要原文作为整体保留或省略，不截断后确认身份。
- Reader 在原有文本之外最多加入 1200 字符观察，原有上下文窗口、输出预算、OCR 预算和图像输入策略不变。实际输入仍通过原有 token 检查。
- 展示文本明确标记为模型观察，不是真值。节点/ECV 的 `observation_ids` 和 Reader 的 `metadata.observation_ids` 记录实际展示 ID；Reader 的 `query_added_observations` 为 0。

checkpoint 绑定观察契约、编码协议、embedding 身份和 memory view。priors 继续绑定 state/index 摘要；node/ECV 绑定观察快照和渲染协议；选择、剪枝记录输入摘要，Reader 绑定 checkpoint、选择记录、问题文件和渲染协议；judge 已按预测行摘要判断缓存。

不兼容缓存会明确拒绝续跑，请使用新产物路径重建受影响阶段。更换 embedding 不要求重做观察；修改源文件或提取设置则必须先生成新的观察快照。查询阶段不会调用提取或 grounding，不写观察，也不更新索引。

## 验证与效果对照

```bash
PYTHONPATH=src:scripts python -m pytest -q tests
python scripts/run_graphmemix_release.py validate-config
```

`tests/test_observation_integration.py` 覆盖无问题文件预处理、来源校验、非方形和 EXIF 图像、重复观察召回、多个问题向量下的预取、canonical vectors、预算、可见范围、checkpoint 恢复、实际 node/ECV/Reader 消息、缓存拒绝及只读边界。模型调用使用确定性测试替身，向量保存和检索使用真实 FAISS。

实际模型质量、Recall@200、候选 Recall@48、准确率和成本仍需在同一数据、模型、记忆视图、候选预算、上下文窗口和冻结系数下做 A/B 对照。观察描述中的错误会影响召回和证据判断，架构回归通过不代表效果提升。

## 本地 Slurm 实测

当前集群可用 `scripts/prepare_observation_gpu_smoke.py` 准备固定的小样本与源码副本，再提交 `scripts/observation_gpu_smoke.sbatch`。该脚本依赖本工作区的 `cluster/cuhk-suite/{token_proxy.py,check_server.py,process_lifecycle.sh}` 和已有本地模型、Conda 环境；它不是发布包的通用部署入口。

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

审计检查观察与原始来源的契约、同一 FAISS 索引内的观察表示、记忆级 ID、canonical graph rows、各阶段覆盖率、真实 HTTP 消息中的观察，以及查询期间离线文件和索引逐字节不变。`structural_checks_passed` 仅表示这些检查通过，不表示每条模型观察语义正确。

2026-10-19 在 gpu03 的 3 张 RTX 4090 D 上完成两轮实测，作业 78875、78876 均正常退出，分别耗时 5 分 58 秒、5 分 42 秒。当前实现保留 39 条观察；试验性的更严格提取/grounding 提示词保留 22 条、拒绝 14 条。两轮都完成 6 个问题的全部阶段和结构审计，关闭服务后的预处理与 Reader 缓存续跑也通过。

人工抽查发现：模型将课堂图片上不清晰的黑板字样断言为 `CHEMISTRY`，模型 grounding 仍接受该观察。更严格提示词的复测仍未排除该误读，且 MemEye 的“最早广告品牌”问题出现节点评分、剪枝丢失正确证据的回归，答案从 Coca-Cola 变成 Burger King。因此未采用该提示词试改，当前代码保持 `come-offline-observe-ground-1`；试验源码与产物单独保存在 `.runtime-observations-gpu-20261019-v2/`。上述质量问题尚未解决。当前实现的实测产物位于 `.runtime-observations-gpu-20261019/`。
