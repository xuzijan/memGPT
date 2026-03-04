#!/usr/bin/env python3
"""
下载 MemGPT 论文官方 benchmark 数据
来源: https://huggingface.co/MemGPT
"""
import os
import json
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "memgpt_benchmark")


def download_qa_data(split="train", limit=None):
    """下载 qa_data (18.6k 条，QA 格式，含 question/answers/ctxs)"""
    from datasets import load_dataset

    print("下载 MemGPT/qa_data...")
    ds = load_dataset("MemGPT/qa_data", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"qa_data_{split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            def _val(x):
                if isinstance(x, dict) and "value" in x:
                    return x["value"]
                return x

            question = _val(row.get("question", ""))
            answers = _val(row.get("answers", []))
            ctxs = _val(row.get("ctxs", []))

            if isinstance(answers, str):
                try:
                    answers = json.loads(answers) if answers.startswith("[") else [answers]
                except Exception:
                    answers = [answers]

            rec = {"id": i, "question": question, "answers": answers, "ctxs": ctxs}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"已保存: {out_path} ({len(ds)} 条)")
    return out_path


def download_sec_filings(limit=100):
    """下载 example-sec-filings (60k 条，SEC 10-K 文档)"""
    from datasets import load_dataset

    print("下载 MemGPT/example-sec-filings...")
    ds = load_dataset("MemGPT/example-sec-filings", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "example_sec_filings.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            rec = {"id": i, **{k: v for k, v in row.items()}}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"已保存: {out_path} ({len(ds)} 条)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["qa_data", "sec_filings", "all"], default="qa_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10, help="qa_data 取前 N 条用于快速验证")
    parser.add_argument("--sec-limit", type=int, default=5, help="sec_filings 取前 N 条")
    args = parser.parse_args()

    if args.dataset in ("qa_data", "all"):
        download_qa_data(split=args.split, limit=args.limit)
    if args.dataset in ("sec_filings", "all"):
        download_sec_filings(limit=args.sec_limit)


if __name__ == "__main__":
    main()
