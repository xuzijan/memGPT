#!/usr/bin/env python3
"""
Mock OpenAI 兼容 API - 用于无真实密钥的流程验证
返回固定响应，不调用真实 API
"""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = 19999
MOCK_RESPONSE = "我是文档分析助手，擅长从长文档中提取要点。（Mock 响应，验证通过）"
MOCK_EMBEDDING = [0.0] * 1536  # text-embedding-3-small dim


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[Mock] {args[0]}")

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _chat_completions(self):
        data = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": MOCK_RESPONSE},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        self._json(data)

    def _embeddings(self):
        data = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": MOCK_EMBEDDING, "index": 0}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        self._json(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if "/chat/completions" in path:
            self._chat_completions()
        elif "/embeddings" in path:
            self._embeddings()
        else:
            self.send_error(404, f"Not found: {path}")

    def do_GET(self):
        if "/health" in self.path or self.path == "/":
            self._json({"status": "ok"})
        else:
            self.send_error(404)


def run(port=PORT):
    server = HTTPServer(("127.0.0.1", port), MockOpenAIHandler)
    print(f"Mock OpenAI 服务: http://127.0.0.1:{port}/v1")
    print("按 Ctrl+C 停止")
    server.serve_forever()


if __name__ == "__main__":
    import sys
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run(p)
