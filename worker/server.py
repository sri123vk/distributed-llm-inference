"""HTTP API for the reference worker. Run: python3 -m worker.server"""
import argparse
import json
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import Engine, CapacityError


def handler_for(engine, info=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                self._json(200, {"ok": True})
            elif self.path == "/stats":
                self._json(200, engine.stats())
            elif self.path == "/info":
                self._json(200, info or {"backend": "reference", "device": "cpu"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/completions":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 16384:
                    raise ValueError("request body must be 1 to 16384 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                stream = body.get("stream", False)
                if not isinstance(stream, bool):
                    raise ValueError("stream must be boolean")
                options = {}
                if info and info.get('backend') == 'llama':
                    chat = body.get('chat', True)
                    if not isinstance(chat, bool):
                        raise ValueError('chat must be boolean')
                    options['chat'] = chat
                request = engine.submit(body.get("prompt"), body.get("max_tokens", 32), **options)
            except CapacityError as exc:
                self._json(503, {"error": str(exc)})
                return
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    while True:
                        event = request.events.get(timeout=120)
                        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                        self.wfile.flush()
                        if event.get("done"):
                            break
                except (BrokenPipeError, ConnectionResetError, queue.Empty):
                    request.cancelled = True
                return
            try:
                while True:
                    event = request.events.get(timeout=120)
                    if "error" in event:
                        self._json(503, {"error": event["error"]})
                        return
                    if event.get("done"):
                        self._json(200, {"id": request.id, "text": "".join(request.output),
                                         "tokens": request.produced})
                        return
            except queue.Empty:
                request.cancelled = True
                self._json(504, {"error": "generation timed out"})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument('--backend', choices=['llama', 'reference'], default='llama')
    parser.add_argument('--model', default='models/smollm2-135m-instruct')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--attention', choices=['torch', 'cuda'], default='torch')
    parser.add_argument('--policy', choices=['continuous', 'static'], default='continuous')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=8)
    args = parser.parse_args()
    if args.backend == 'llama':
        import torch
        from .llama import load_pretrained
        from .batched import BatchedEngine
        torch.set_num_threads(args.threads)
        model = load_pretrained(args.model, args.device, args.blocks, args.block_size, args.attention)
        engine = BatchedEngine(model, args.max_batch, args.policy)
    else:
        engine = Engine(args.blocks, args.block_size, args.max_batch)
    info = {'backend': args.backend, 'model': args.model if args.backend == 'llama' else 'synthetic',
            'device': args.device, 'attention': args.attention, 'policy': args.policy}
    engine.start()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(engine, info))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.stop()


if __name__ == "__main__":
    main()
