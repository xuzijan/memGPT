#!/usr/bin/env python3
"""
MemGPT/Letta 最小示例

模式 1 - Mock 验证（无需任何密钥）:
  python run_minimal_example.py --validate-mock

模式 2 - 指定 YAML 配置:
  python run_minimal_example.py --config experiments/configs/memgpt_baseline.yaml

模式 3 - Letta Cloud / 本地 Server / OpenAI
"""
import os
import sys

# 加载 .env（若存在）
try:
    from dotenv import load_dotenv
    _base = os.path.dirname(os.path.abspath(__file__))
    for p in [
        os.path.join(_base, ".env"),
        os.path.join(_base, "..", ".env"),
        os.path.join(_base, "..", "..", ".env"),
        os.path.join(_base, "..", "..", "memGPT", ".env"),
        os.path.expanduser("~/.letta/.env"),
    ]:
        if os.path.isfile(p):
            load_dotenv(p)
            break
except ImportError:
    pass


def load_config(config_path=None):
    """从 YAML 加载配置，未指定时使用默认路径"""
    try:
        import yaml
    except ImportError:
        return {}
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_script_dir))
    default_path = os.path.join(_root, "experiments", "configs", "memgpt_baseline.yaml")
    path = config_path or os.getenv("EXPERIMENT_CONFIG") or default_path
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_letta_cloud(config=None):
    """使用 Letta Cloud API"""
    from letta_client import Letta

    api_key = os.getenv("LETTA_API_KEY")
    if not api_key:
        print("错误: 需要设置 LETTA_API_KEY（在 https://app.letta.com 获取）")
        return False

    client = Letta(api_key=api_key)
    return _run_with_client(client, "Letta Cloud", config or {})


def run_letta_local(config=None):
    """使用本地 Letta Server"""
    from letta_client import Letta

    base_url = os.getenv("LETTA_SERVER_URL", "http://localhost:8283")
    client = Letta(base_url=base_url)
    return _run_with_client(client, f"本地 Server ({base_url})", config or {})


def _run_with_client(client, mode: str, config: dict):
    """通用 Agent 创建与对话逻辑，参数从 config 读取"""
    print("=== MemGPT/Letta 最小示例 ===\n")
    print(f"模式: {mode}\n")

    model = config.get("model", {})
    mb = config.get("memory_blocks", {})
    prompt = config.get("prompt", {})

    model_handle = model.get("handle", "openai/gpt-4o-mini")
    embedding = model.get("embedding", "openai/text-embedding-3-small")
    memory_blocks = [
        {"label": "persona", "value": mb.get("persona", "我是文档分析助手，擅长从长文档中提取要点。")},
        {"label": "human", "value": mb.get("human", "用户正在做 MemGPT 研究复现。")},
    ]
    user_msg = prompt.get("user", "你好，请介绍一下你自己")

    try:
        print("创建 Agent...")
        agent_state = client.agents.create(
            model=model_handle,
            embedding=embedding,
            memory_blocks=memory_blocks,
        )
        print(f"Agent 已创建: {agent_state.id}\n")

        print(f"发送消息: {user_msg}")
        response = client.agents.messages.create(
            agent_id=agent_state.id,
            input=user_msg,
        )

        print("\n--- Agent 回复 ---")
        for msg in response.messages:
            if hasattr(msg, "content") and msg.content:
                if isinstance(msg.content, list):
                    for c in msg.content:
                        if hasattr(c, "text"):
                            print(c.text)
                        elif isinstance(c, str):
                            print(c)
                else:
                    print(msg.content)
            elif hasattr(msg, "text"):
                print(msg.text)
            else:
                print(msg)

        print("\n=== 示例运行完成 ===")
        return True

    except Exception as e:
        print(f"\n错误: {e}")
        return False


def run_openai_fallback(base_url=None, api_key=None, config=None):
    """纯 OpenAI 兜底：无记忆，验证 API 可用（支持伪密钥 + 自定义 Base URL）"""
    try:
        from openai import OpenAI
    except ImportError:
        print("错误: 需要安装 openai 包 (pip install openai)")
        return False

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        print("错误: 需要设置 OPENAI_API_KEY")
        return False

    cfg = config or {}
    model = cfg.get("model", {})
    prompt = cfg.get("prompt", {})

    model_handle = model.get("handle", "openai/gpt-4o-mini")
    max_tokens = model.get("max_tokens", 4096)
    system_msg = prompt.get("system", "你是文档分析助手。请用一句话介绍自己。")
    user_msg = prompt.get("user", "你好，请介绍一下你自己")

    is_mock = base_url or os.getenv("OPENAI_BASE_URL")
    print("=== OpenAI 示例 ===\n")
    print(f"Base URL: {base_url or os.getenv('OPENAI_BASE_URL') or '默认'}\n")

    client = OpenAI(api_key=key, base_url=base_url or os.getenv("OPENAI_BASE_URL"))
    resp = client.chat.completions.create(
        model=model_handle.split("/")[-1] if "/" in model_handle else model_handle,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=min(max_tokens, 200),
    )
    reply = resp.choices[0].message.content
    print("--- 回复 ---")
    print(reply)
    print("\n=== 环境验证完成 ===")
    if not is_mock:
        print("提示: 完整 MemGPT 功能需 Letta Cloud (LETTA_API_KEY) 或本地 Server (PostgreSQL)")
    return True


def run_mock_validation(config=None):
    """伪密钥 + Mock Base URL：无需真实 API，验证调用链路"""
    import threading
    import time

    from http.server import HTTPServer

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from mock_openai_server import MockOpenAIHandler, PORT

    server = HTTPServer(("127.0.0.1", PORT), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)  # 等待启动

    try:
        os.environ["OPENAI_API_KEY"] = "sk-dummy"
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
        print("=== Mock 验证模式（伪密钥 + 自定义 Base URL）===\n")
        print(f"Mock 服务: http://127.0.0.1:{PORT}/v1")
        print("API Key: sk-dummy\n")
        return run_openai_fallback(
            base_url=f"http://127.0.0.1:{PORT}/v1",
            api_key="sk-dummy",
            config=config or load_config(),
        )
    finally:
        server.shutdown()


def main():
    # 解析 --config 参数
    config_path = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    config = load_config(config_path)
    if config and config_path:
        print(f"已加载配置: {config_path}\n")

    if "--validate-mock" in sys.argv or os.getenv("VALIDATE_MOCK"):
        ok = run_mock_validation(config)
    elif os.getenv("LETTA_API_KEY"):
        ok = run_letta_cloud(config)
    elif os.getenv("LETTA_SERVER_URL"):
        ok = run_letta_local(config)
    elif os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_API_KEY"):
        ok = run_openai_fallback(config=config)
    elif os.getenv("OPENAI_API_KEY"):
        ok = run_openai_fallback(config=config)
    else:
        print("未检测到有效配置。可选：")
        print("  --validate-mock    伪密钥 + Mock，无需真实 API")
        print("  OPENAI_BASE_URL + OPENAI_API_KEY  自定义 Base URL（可伪密钥）")
        print("  OPENAI_API_KEY     真实 OpenAI 验证")
        print("  LETTA_API_KEY      Letta Cloud")
        sys.exit(1)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
