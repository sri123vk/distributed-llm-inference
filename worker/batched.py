"""Continuous/static batch policies over the same trained-model execution path."""
from collections import deque
from dataclasses import dataclass, field
import threading
import time

from .engine import Request, CapacityError


@dataclass
class ModelRequest(Request):
    prompt_ids: list = field(default_factory=list)
    generated_ids: list = field(default_factory=list)
    submitted_at: float = field(default_factory=time.monotonic)
    first_token_at: float = 0.0
    text: str = ''
    ignore_eos: bool = False
    prefix_offset: int = 0
    read_offset: int = 0


def detokenize(tokenizer, ids, prefix_offset, read_offset, final=False):
    """Incremental detokenization: decode only a short trailing window per token.

    Returns (piece, prefix_offset, read_offset). Text is held back while the window
    ends in an incomplete multibyte character, unless this is the final token.
    """
    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    prefix = decode(ids[prefix_offset:read_offset])
    text = decode(ids[prefix_offset:])
    if len(text) > len(prefix) and (final or not text.endswith('\ufffd')):
        return text[len(prefix):], read_offset, len(ids)
    return '', prefix_offset, read_offset


class BatchedEngine:
    def __init__(self, model, max_batch=8, policy='continuous', max_queue=128,
                 max_context=1024, prefill_budget=1024):
        if policy not in ('continuous', 'static') or max_batch < 1:
            raise ValueError('invalid batch policy or size')
        self.model, self.kv = model, model.cache
        self.max_batch, self.policy = max_batch, policy
        self.max_queue, self.max_context = max_queue, max_context
        self.prefill_budget = prefill_budget
        self.waiting = deque()
        self.active = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.completed = self.rejected = self.generated_tokens = 0
        self.prefill_batches = self.decode_batches = self.max_observed_batch = 0
        eos = model.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.eos_ids.add(model.tokenizer.eos_token_id)

    def submit(self, prompt, max_tokens, chat=True, ignore_eos=False):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
            raise ValueError('prompt must contain 1 to 16384 characters')
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 256:
            raise ValueError('max_tokens must be between 1 and 256')
        if chat:
            ids = self.model.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
        else:
            ids = self.model.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids or len(ids) + max_tokens > min(self.max_context, self.model.config.max_position_embeddings):
            raise ValueError('prompt plus generation exceeds context limit')
        if len(ids) > self.prefill_budget:
            raise ValueError('prompt exceeds per-step prefill budget')
        if len(ids) + max_tokens > self.kv.total_blocks * self.kv.block_size:
            raise ValueError('request cannot fit in the KV cache')
        request = ModelRequest(prompt=prompt, max_tokens=max_tokens, prompt_ids=ids, ignore_eos=ignore_eos)
        with self.lock:
            if self.stop_event.is_set():
                raise CapacityError('worker is shutting down')
            if len(self.waiting) >= self.max_queue:
                self.rejected += 1
                raise CapacityError('request queue is full')
            self.waiting.append(request)
        return request

    def _finish(self, request, completed=True):
        self.kv.release(request.id)
        self.active.pop(request.id, None)
        self.completed += completed

    def tick(self):
        # Only the engine thread mutates active requests and the cache.
        for request in list(self.active.values()):
            if request.cancelled:
                request.events.put({'done': True, 'cancelled': True})
                self._finish(request)
        admitted = []
        budget = self.prefill_budget
        with self.lock:
            can_admit = self.policy == 'continuous' or not self.active
            while can_admit and self.waiting and len(self.active) + len(admitted) < self.max_batch:
                request = self.waiting[0]
                if request.cancelled:
                    self.waiting.popleft()
                    request.events.put({'done': True, 'cancelled': True})
                    continue
                if len(request.prompt_ids) > budget:
                    break
                if not self.kv.reserve(request.id, len(request.prompt_ids) + request.max_tokens):
                    break  # Backpressure: wait for an active request to release pages.
                self.waiting.popleft()
                budget -= len(request.prompt_ids)
                admitted.append(request)
        if admitted:
            try:
                logits = self.model.prefill([r.id for r in admitted], [r.prompt_ids for r in admitted])
                next_tokens = logits.argmax(dim=-1).tolist()
                self.prefill_batches += 1
                for request, token in zip(admitted, next_tokens):
                    request.last_token = token
                    request.admitted_at = time.monotonic()
                    self.active[request.id] = request
            except Exception as exc:
                for request in admitted:
                    self.kv.release(request.id)
                    request.events.put({'error': str(exc), 'done': True})
        self.max_observed_batch = max(self.max_observed_batch, len(self.active))
        for request in list(self.active.values()):
            token = request.last_token
            request.generated_ids.append(token)
            request.produced += 1
            self.generated_tokens += 1
            if not request.first_token_at:
                request.first_token_at = time.monotonic()
            stopped = token in self.eos_ids and not request.ignore_eos
            done = stopped or request.produced >= request.max_tokens
            piece, request.prefix_offset, request.read_offset = detokenize(
                self.model.tokenizer, request.generated_ids, request.prefix_offset,
                request.read_offset, final=done)
            if piece:
                request.text += piece
                request.output.append(piece)
            request.events.put({'token': piece, 'token_id': token, 'done': done,
                'finish_reason': ('stop' if stopped else 'length') if done else None,
                'ttft_ms': round((request.first_token_at - request.submitted_at) * 1000, 3)})
            if done:
                self._finish(request)
        remaining = list(self.active.values())
        if remaining:
            try:
                logits = self.model.decode([r.id for r in remaining], [r.last_token for r in remaining])
                self.decode_batches += 1
                for request, token in zip(remaining, logits.argmax(dim=-1).tolist()):
                    request.last_token = token
            except Exception as exc:
                for request in remaining:
                    request.events.put({'error': str(exc), 'done': True})
                    self._finish(request, completed=False)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self.tick()
                if not self.active:
                    self.stop_event.wait(.002)
        finally:
            with self.lock:
                pending = list(self.waiting) + list(self.active.values())
                self.waiting.clear()
            for request in pending:
                request.events.put({'error': 'worker shutting down', 'done': True})
                self.kv.release(request.id)
            self.active.clear()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=30)

    def stats(self):
        with self.lock:
            return {'active': len(self.active), 'queued': len(self.waiting),
                'free_blocks': len(self.kv.free), 'total_blocks': self.kv.total_blocks,
                'completed': self.completed, 'rejected': self.rejected,
                'generated_tokens': self.generated_tokens, 'prefill_batches': self.prefill_batches,
                'decode_batches': self.decode_batches, 'max_observed_batch': self.max_observed_batch}
