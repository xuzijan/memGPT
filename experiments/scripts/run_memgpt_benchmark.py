#!/usr/bin/env python3
"""
MemGPT 论文 benchmark 跑通脚本
使用 qa_data：将 ctxs 作为 archival memory，用 question 查询，与 answers 对比

需要: LETTA_API_KEY 或 OPENAI_API_KEY + 本地 Server
或: --validate-mock 用 Mock 快速验证流程
"""
import os
import sys
import json
import argparse

# 加载 .env
try:
    from dotenv import load_dotenv
    _base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(_base, "..", ".env"), os.path.join(_base, "..", "..", ".env")]:
        if os.path.isfile(p):
            load_dotenv(p)
            break
except ImportError:
    pass


def load_config(config_path=None):
    """加载 YAML 配置"""
    try:
        import yaml
    except ImportError:
        return {}
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(_root, "experiments", "configs", "memgpt_baseline.yaml")
    path = config_path or os.getenv("EXPERIMENT_CONFIG") or default_path
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_qa_data(path, limit=5):
    """加载 qa_data JSONL"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def run_with_letta(samples, config):
    """使用 Letta API 运行（需 LETTA_API_KEY）"""
    from letta_client import Letta

    api_key = os.getenv("LETTA_API_KEY")
    if not api_key:
        print("错误: 需要 LETTA_API_KEY")
        return False

    model = config.get("model", {})
    client = Letta(api_key=api_key)

    results = []
    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        ctxs = sample.get("ctxs", [])

        # 将 ctxs 拼成 archival 文本（MemGPT 会存入 archival memory）
        context_text = "\n\n".join(
            f"[{c.get('title','')}]\n{c.get('text','')}" for c in (ctxs[:5] if isinstance(ctxs, list) else [])
        )

        # 创建 agent，将 context 放入 human 或单独 memory
        memory_blocks = [
            {"label": "persona", "value": "你是文档分析助手。根据提供的上下文回答问题。"},
            {"label": "human", "value": f"上下文:\n{context_text[:8000]}"},  # 截断避免超长
        ]

        agent_state = client.agents.create(
            model=model.get("handle", "openai/gpt-4o-mini"),
            embedding=model.get("embedding", "openai/text-embedding-3-small"),
            memory_blocks=memory_blocks,
        )

        response = client.agents.messages.create(agent_id=agent_state.id, input=question)
        pred = ""
        for msg in response.messages:
            if hasattr(msg, "content") and msg.content:
                if isinstance(msg.content, list):
                    for c in msg.content:
                        if hasattr(c, "text"):
                            pred = (pred + " " + c.text).strip()
                else:
                    pred = str(msg.content)

        correct = any(a.lower() in pred.lower() for a in answers) if answers else None
        results.append({"id": i, "question": question, "pred": pred, "answers": answers, "correct": correct})
        print(f"[{i+1}/{len(samples)}] Q: {question[:50]}... -> {pred[:80]}... (correct={correct})")

    return results


def run_with_openai(samples, config):
    """使用 OpenAI 直接调用（无 archival，仅验证流程）"""
    try:
        from openai import OpenAI
    except ImportError:
        print("错误: pip install openai")
        return []

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("错误: 需要 OPENAI_API_KEY")
        return []

    model = config.get("model", {})
    client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL"))

    results = []
    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        ctxs = sample.get("ctxs", [])

        context_text = "\n\n".join(
            f"[{c.get('title','')}]\n{c.get('text','')}" for c in (ctxs[:3] if isinstance(ctxs, list) else [])
        )[:4000]

        resp = client.chat.completions.create(
            model=model.get("handle", "gpt-4o-mini").split("/")[-1],
            messages=[
                {"role": "system", "content": "根据上下文回答问题，只输出答案。"},
                {"role": "user", "content": f"上下文:\n{context_text}\n\n问题: {question}"},
            ],
            max_tokens=200,
        )
        pred = resp.choices[0].message.content or ""
        correct = any(a.lower() in pred.lower() for a in answers) if answers else None
        results.append({"id": i, "question": question, "pred": pred, "answers": answers, "correct": correct})
        print(f"[{i+1}/{len(samples)}] Q: {question[:50]}... -> {pred[:80]}... (correct={correct})")

    return results


def run_mock(samples, config):
    """Mock 模式：固定返回，仅验证流程"""
    results = []
    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        pred = "[Mock] " + (answers[0] if answers else "N/A")
        correct = True if answers else None
        results.append({"id": i, "question": question, "pred": pred, "answers": answers, "correct": correct})
        print(f"[{i+1}/{len(samples)}] Q: {question[:50]}... -> {pred[:80]}... (correct={correct})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="qa_data JSONL 路径")
    parser.add_argument("--limit", type=int, default=3, help="运行样本数")
    parser.add_argument("--validate-mock", action="store_true", help="Mock 模式")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    data_path = args.data or os.path.join(
        os.path.dirname(__file__), "..", "data", "memgpt_benchmark", "qa_data_train.jsonl"
    )
    if not os.path.isfile(data_path):
        print(f"数据不存在: {data_path}")
        print("请先运行: python download_memgpt_benchmark.py --limit 10")
        print("(网络不可用时，可手动创建该 JSONL 文件，格式见 MemGPT/qa_data)")
        sys.exit(1)

    samples = load_qa_data(data_path, limit=args.limit)
    print(f"加载 {len(samples)} 条样本\n")

    if args.validate_mock:
        results = run_mock(samples, config)
    elif os.getenv("LETTA_API_KEY"):
        results = run_with_letta(samples, config)
    elif os.getenv("OPENAI_API_KEY"):
        results = run_with_openai(samples, config)
    else:
        print("请设置 LETTA_API_KEY 或 OPENAI_API_KEY，或使用 --validate-mock")
        sys.exit(1)

    # 统计
    if results:
        correct_count = sum(1 for r in results if r.get("correct") is True)
        total = len(results)
        print(f"\n=== 结果 ===\n正确: {correct_count}/{total} ({100*correct_count/total:.1f}%)")


if __name__ == "__main__":
    main()
