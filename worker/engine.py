"""Dependency-free reference engine for continuous batching and paged attention.

The deterministic byte model is a systems test backend, not a trained LLM.
"""
from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class CapacityError(Exception):
    pass


class PagedKV:
    def __init__(self, blocks: int = 1024, block_size: int = 16):
        if blocks < 1 or block_size < 1:
            raise ValueError("blocks and block_size must be positive")
        self.block_size = block_size
        self.free = list(range(blocks - 1, -1, -1))
        self.keys: List[List[Optional[List[float]]]] = [[None] * block_size for _ in range(blocks)]
        self.values: List[List[Optional[List[float]]]] = [[None] * block_size for _ in range(blocks)]
        self.tables: Dict[str, List[int]] = {}
        self.lengths: Dict[str, int] = {}

    def reserve(self, request_id: str, tokens: int) -> bool:
        """Atomically reserve enough pages for the entire request."""
        if request_id in self.tables:
            raise ValueError("duplicate request")
        needed = (tokens + self.block_size - 1) // self.block_size
        if needed > len(self.free):
            return False
        self.tables[request_id] = [self.free.pop() for _ in range(needed)]
        self.lengths[request_id] = 0
        return True

    def append(self, request_id: str, key: List[float], value: List[float]) -> None:
        n = self.lengths[request_id]
        table = self.tables[request_id]
        if n >= len(table) * self.block_size:
            raise CapacityError("reserved KV pages exhausted")
        block = table[n // self.block_size]
        offset = n % self.block_size
        self.keys[block][offset] = key
        self.values[block][offset] = value
        self.lengths[request_id] = n + 1

    def attention(self, request_id: str, query: List[float]) -> List[float]:
        n = self.lengths[request_id]
        if n == 0:
            return [0.0] * len(query)
        scale = 1.0 / math.sqrt(len(query))
        scores = []
        vals = []
        for i in range(n):
            block = self.tables[request_id][i // self.block_size]
            offset = i % self.block_size
            key = self.keys[block][offset]
            value = self.values[block][offset]
            scores.append(sum(a * b for a, b in zip(query, key)) * scale)
            vals.append(value)
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        denominator = sum(weights)
        return [sum(w * v[d] for w, v in zip(weights, vals)) / denominator for d in range(len(query))]

    def release(self, request_id: str) -> None:
        for block in self.tables.pop(request_id, []):
            self.keys[block] = [None] * self.block_size
            self.values[block] = [None] * self.block_size
            self.free.append(block)
        self.lengths.pop(request_id, None)


class ReferenceModel:
    """Small deterministic attention model used to exercise the serving path."""

    width = 16
    vocabulary = tuple(range(32, 127))

    def embedding(self, token: int) -> List[float]:
        return [math.sin((token + 1) * (i + 1) * 0.071) for i in range(self.width)]

    def step(self, kv: PagedKV, request_id: str, token: int) -> int:
        query = self.embedding(token)
        key = [math.cos(x + i * 0.13) for i, x in enumerate(query)]
        value = [math.tanh(x * 1.7 + i * 0.01) for i, x in enumerate(query)]
        kv.append(request_id, key, value)
        context = kv.attention(request_id, query)
        return max(self.vocabulary, key=lambda candidate: sum(
            a * b for a, b in zip(context, self.embedding(candidate))))


@dataclass
class Request:
    prompt: str
    max_tokens: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: bool = False
    produced: int = 0
    last_token: int = 32
    output: List[str] = field(default_factory=list)
    admitted_at: float = 0.0


class Engine:
    def __init__(self, blocks: int = 1024, block_size: int = 16, max_batch: int = 16,
                 tick_seconds: float = 0.005):
        self.kv = PagedKV(blocks, block_size)
        self.model = ReferenceModel()
        self.max_batch = max_batch
        self.tick_seconds = tick_seconds
        self.waiting: queue.Queue[Request] = queue.Queue()
        self.active: Dict[str, Request] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.completed = 0
        self.rejected = 0

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def submit(self, prompt: str, max_tokens: int) -> Request:
        if not isinstance(prompt, str) or not prompt or len(prompt) > 4096:
            raise ValueError("prompt must contain 1 to 4096 characters")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 256:
            raise ValueError("max_tokens must be between 1 and 256")
        request = Request(prompt=prompt, max_tokens=max_tokens)
        self.waiting.put(request)
        return request

    def _admit(self) -> None:
        while len(self.active) < self.max_batch:
            try:
                request = self.waiting.get_nowait()
            except queue.Empty:
                break
            if request.cancelled:
                request.events.put({"done": True, "cancelled": True})
                continue
            prompt_tokens = list(request.prompt.encode("utf-8"))
            if not self.kv.reserve(request.id, len(prompt_tokens) + request.max_tokens):
                self.rejected += 1
                request.events.put({"error": "KV cache capacity exhausted", "done": True})
                continue
            request.admitted_at = time.monotonic()
            try:
                for token in prompt_tokens:
                    request.last_token = self.model.step(self.kv, request.id, token)
            except Exception as exc:
                self.kv.release(request.id)
                request.events.put({"error": str(exc), "done": True})
                continue
            self.active[request.id] = request

    def tick(self) -> None:
        """Admit new requests, then decode one token per active request."""
        with self.lock:
            self._admit()
            for request in list(self.active.values()):
                if request.cancelled:
                    request.events.put({"done": True, "cancelled": True})
                    self._finish(request)
                    continue
                try:
                    token = request.last_token
                    request.produced += 1
                    piece = chr(token)
                    request.output.append(piece)
                    done = request.produced >= request.max_tokens
                    request.events.put({"token": piece, "done": done})
                    if done:
                        self._finish(request)
                    else:
                        request.last_token = self.model.step(self.kv, request.id, token)
                except Exception as exc:
                    request.events.put({"error": str(exc), "done": True})
                    self._finish(request)

    def _finish(self, request: Request) -> None:
        self.kv.release(request.id)
        self.active.pop(request.id, None)
        self.completed += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.tick()
            self.stop_event.wait(self.tick_seconds)

    def stats(self) -> dict:
        with self.lock:
            return {"active": len(self.active), "queued": self.waiting.qsize(),
                    "free_blocks": len(self.kv.free), "total_blocks": len(self.kv.keys),
                    "completed": self.completed, "rejected": self.rejected}
