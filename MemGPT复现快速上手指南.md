# MemGPT 复现快速上手指南

> 按 6 步递进：Fork → 环境隔离 → 跑通示例 → 整理 I/O 与 YAML → 统一评估与 AmbiCoding

---

## 论文背景

**论文**：*MemGPT: Towards LLMs as Operating Systems* (arXiv:2310.08560)  
**机构**：UC Berkeley Sky Computing Lab  
**核心思想**：用虚拟上下文管理（类似 OS 的分页）让 LLM 在有限 context 与外部存储间「换页」，实现近似无限上下文。

**评估场景**：文档分析、多轮对话  

**当前项目**：`xuzijan/memGPT` 是 `letta-ai/letta` 的 fork。

---

## Step 1：Fork 到自己仓库

- [x] Fork `letta-ai/letta` → `xuzijan/memGPT`（或你的 GitHub 用户名）
- [ ] 在 README 或 commit 中注明 fork 来源与对应 commit
- [ ] `.gitignore` 排除大文件（模型权重、数据等），只同步代码和配置

**建议 .gitignore 新增：**

```gitignore
*.safetensors
*.bin
*.pt
*.pth
models/
checkpoints/
.cache/huggingface/
```

---

## Step 2：环境隔离

每个 baseline 单独环境，避免依赖冲突。

**目录结构：**

```
/root/autodl-tmp/
├── baselines/
│   └── memgpt/              # 论文 1
│       ├── .venv/           # 或 conda env
│       ├── memGPT/          # 上游代码（或软链）
│       └── requirements.txt
├── experiments/
│   ├── configs/
│   ├── data/
│   ├── eval/
│   └── outputs/
└── ...
```

**MemGPT 环境：**

```bash
cd /root/autodl-tmp/memGPT
python -m venv .venv
source .venv/bin/activate  # Linux
pip install -e ".[sqlite,server]"
pip install asyncpg python-dotenv
```

**配置 API Key：**

```bash
cp experiments/.env.example experiments/.env
# 编辑填入 OPENAI_API_KEY=sk-xxx 或 LETTA_API_KEY=xxx
```

---

## Step 3：按原作者示例跑通一次

### 3.1 Mock 验证（无需密钥）

```bash
cd /root/autodl-tmp/experiments/scripts
python run_minimal_example.py --validate-mock
```

### 3.2 真实 API 验证

```bash
# 方式 A：Letta Cloud
export LETTA_API_KEY=xxx
python run_minimal_example.py

# 方式 B：OpenAI
export OPENAI_API_KEY=sk-xxx
python run_minimal_example.py
```

### 3.3 本地 Server（需 Docker + PostgreSQL）

```bash
cd memGPT
docker compose up -d
export LETTA_SERVER_URL=http://localhost:8283
python experiments/scripts/run_minimal_example.py
```

### 3.4 论文官方 benchmark 跑通

**数据来源**：Hugging Face [MemGPT/qa_data](https://huggingface.co/datasets/MemGPT/qa_data)、[MemGPT/example-sec-filings](https://huggingface.co/datasets/MemGPT/example-sec-filings)

**下载数据**（需网络）：
```bash
python experiments/scripts/download_memgpt_benchmark.py --limit 10
```

**运行 benchmark**：
```bash
# Mock 验证（无需 API）
python experiments/scripts/run_memgpt_benchmark.py --validate-mock --limit 3

# 真实 API（需 LETTA_API_KEY 或 OPENAI_API_KEY）
python experiments/scripts/run_memgpt_benchmark.py --limit 5
```

**离线样本**：`experiments/data/memgpt_benchmark/qa_data_train.jsonl` 含 3 条示例，格式与论文 qa_data 一致。

---

## Step 4：整理输入输出、重要参数、YAML

### 4.1 输入格式（完整）

**最小示例 `run_minimal_example.py`**

| 输入 | 类型 | 说明 | 来源 |
|------|------|------|------|
| model | str | LLM 句柄 | YAML `model.handle` |
| embedding | str | 嵌入模型 | YAML `model.embedding` |
| memory_blocks | list[dict] | persona、human 等 | YAML `memory_blocks` |
| user message | str | 用户提问 | YAML `prompt.user` 或代码传入 |

**Benchmark `run_memgpt_benchmark.py`**

| 输入 | 类型 | 说明 | 来源 |
|------|------|------|------|
| 数据文件 | JSONL | 每行一条样本 | `--data` 或默认 `qa_data_train.jsonl` |
| question | str | 问题 | 数据字段 |
| answers | list[str] | 标准答案（可多个） | 数据字段 |
| ctxs | list[dict] | 上下文段落 `{title, text}` | 数据字段 |

**qa_data 单条样本格式：**
```json
{"id": 0, "question": "...", "answers": ["a1", "a2"], "ctxs": [{"title": "...", "text": "..."}]}
```

### 4.2 输出格式（完整）

**最小示例**

| 输出 | 类型 | 说明 |
|------|------|------|
| Agent 回复 | str | `response.messages` 中提取的文本 |
| 工具调用 | 可选 | 如 `archival_memory_search`（Letta 模式） |

**Benchmark**

| 输出 | 类型 | 说明 |
|------|------|------|
| results | list[dict] | 每条 `{id, question, pred, answers, correct}` |
| correct | bool | `pred` 中是否包含任一 `answers` |
| 统计 | - | 正确数/总数、准确率 |

**Benchmark 单条结果格式：**
```json
{"id": 0, "question": "...", "pred": "...", "answers": ["..."], "correct": true}
```

### 4.3 重要参数（完整）

| 参数 | 位置 | 默认 | 说明 |
|------|------|------|------|
| temperature | YAML `model.temperature` | 0.7 | 生成随机性，0~2 |
| top_p | YAML `model.top_p` | 1.0 | 核采样（Letta 中部分模型支持） |
| max_tokens | YAML `model.max_tokens` | 4096 | 最大生成长度 |
| handle | YAML `model.handle` | openai/gpt-4o-mini | 模型句柄 |
| embedding | YAML `model.embedding` | openai/text-embedding-3-small | 嵌入模型 |
| seed | YAML `experiment.seed` | 42 | 随机种子 |

### 4.4 YAML 配置文件（完整）

**路径**：`experiments/configs/memgpt_baseline.yaml`

| 字段 | 类型 | 说明 |
|------|------|------|
| `experiment.name` | str | 实验名称 |
| `experiment.seed` | int | 随机种子 |
| `model.handle` | str | LLM 模型句柄，如 `openai/gpt-4o-mini` |
| `model.embedding` | str | 嵌入模型，如 `openai/text-embedding-3-small` |
| `model.temperature` | float | 温度 |
| `model.top_p` | float | Top-P |
| `model.max_tokens` | int | 最大 token 数 |
| `memory_blocks.persona` | str | persona 记忆块内容 |
| `memory_blocks.human` | str | human 记忆块内容 |
| `prompt.system` | str | 系统提示（OpenAI 兜底用） |
| `prompt.user` | str | 默认用户提示 |
| `data.path` | str | 数据文件路径 |
| `data.split` | str | 数据划分 |

**完整示例：**
```yaml
experiment:
  name: memgpt_paper_repro
  seed: 42

model:
  handle: "openai/gpt-4o-mini"
  embedding: "openai/text-embedding-3-small"
  temperature: 0.7
  top_p: 1.0
  max_tokens: 4096

memory_blocks:
  persona: "我是文档分析助手，擅长从长文档中提取要点。"
  human: "用户正在做 MemGPT 研究复现。"

prompt:
  system: "你是文档分析助手。请用一句话介绍自己。"
  user: "你好，请介绍一下你自己"

data:
  path: "experiments/data/memgpt_benchmark/qa_data_train.jsonl"
  split: "test"
```

**加载方式**：`--config path` 或环境变量 `EXPERIMENT_CONFIG`

### 4.5 脚本命令行参数

**run_minimal_example.py**

| 参数 | 说明 |
|------|------|
| `--validate-mock` | Mock 模式，无需 API |
| `--config PATH` | 指定 YAML 配置路径 |

**run_memgpt_benchmark.py**

| 参数 | 说明 |
|------|------|
| `--data PATH` | qa_data JSONL 路径 |
| `--limit N` | 运行样本数 |
| `--validate-mock` | Mock 模式 |
| `--config PATH` | 指定 YAML 配置路径 |

---

## Step 5：统一评估与 AmbiCoding 适配

### 5.1 统一输入输出接口

- **统一输入**：`{id, query, context, options?, ...}` 的 dict
- **统一输出**：`{id, pred, ground_truth?, metadata?}` 的 dict
- **统一入口**：`run_baseline("memgpt", config_path)` 分发到各实现

### 5.2 AmbiCoding 数据集转换

- 明确 AmbiCoding 原始格式
- 为 MemGPT 写转换脚本：`AmbiCoding → MemGPT 输入格式`
- 转换脚本需可复现（固定 seed、版本）

### 5.3 统一评估脚本

- 输入：各 baseline 的预测结果（统一格式）
- 输出：同一套指标（accuracy、exact match 等）
- 评估逻辑与 baseline 解耦

### 5.4 Prompt Template

**论文/代码中的 Prompt**

- 文档分析 persona：`letta/personas/examples/memgpt_doc.txt`  
  核心指令：

> The answer to the human's question will usually be located somewhere in your archival memory, so keep paging through results until you find enough information to construct an answer.

- 待整理：system prompt、user prompt 结构、few-shot 示例、占位符

**建议**：在 YAML 或单独文件中保存完整 prompt 模板，便于复现与对比。

---

## 数据流简图

```
用户输入 → Agent (LLM) → 工具调用(archival_search) → 检索结果 → 再生成 → 最终回复
                ↑
                └── core memory (persona, human) + archival memory (长文档)
```

---

## 常见问题

| 问题 | 排查 |
|------|------|
| 找不到 API Key | 检查 `.env` 或 `OPENAI_API_KEY` |
| 数据库连接失败 | 本地 Server 需 PostgreSQL，见 `compose.yaml` |
| 超参数不生效 | 确认从 YAML 加载并传入 agent 配置 |

---

## 参考链接

- [MemGPT 论文](https://arxiv.org/abs/2310.08560)
- [MemGPT 研究官网](https://research.memgpt.ai/)
- [Letta 文档](https://docs.letta.com/)
