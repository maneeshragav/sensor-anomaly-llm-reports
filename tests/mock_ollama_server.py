"""
A tiny stand-in for Ollama's HTTP API, used ONLY to test llm_report.py's
request/response handling in an environment where the real Ollama server
(running on the user's own machine) isn't reachable. It mimics the exact
response shape Ollama's real /api/generate endpoint returns.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence default logging
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path != "/api/generate":
            self.send_response(404)
            self.end_headers()
            return

        prompt = body.get("prompt", "")
        fake_response = (
            "MOCK RESPONSE (this stands in for a real Ollama model): "
            f"Anomaly report generated from a prompt of {len(prompt)} characters, "
            f"for model '{body.get('model')}'."
        )
        payload = json.dumps({
            "model": body.get("model"),
            "created_at": "2026-01-01T00:00:00Z",
            "response": fake_response,
            "done": True,
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_mock_server(port: int = 11434) -> HTTPServer:
    server = HTTPServer(("localhost", port), MockOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
