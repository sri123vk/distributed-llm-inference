# Distributed LLM inference platform

A small LLM serving stack built from the scheduling layer down: a batched worker that runs a real pretrained model with its own **paged KV cache** and **continuous batching**, a Go **gateway** that load-balances and fails over across worker replicas, a **GPU-aware Kubernetes scheduler extender**, and a **CUDA paged-attention kernel**.

| Component | Language | What it does |
|---|---|---|
| `worker/llama.py` | Python / PyTorch | Prefill and decode for Llama-architecture checkpoints (default: SmolLM2-135M-Instruct). Hugging Face supplies only weights and layer modules; RoPE, grouped-query attention, KV writes and paged attention reads are implemented here. |
| `worker/batched.py` | Python | Continuous or static batching over the same execution path, admission control, backpressure, cancellation, incremental detokenization, streaming events. |
| `cuda/` | CUDA / C++ | Paged attention kernel with GQA (FP32 and FP16) and a PyTorch extension binding, selected with `--attention cuda`. |
| `gateway/` | Go | Least-loaded routing over a cached healthy set, background health checks, retry on another worker, SSE pass-through, dashboard. |
| `scheduler/` | Go | kube-scheduler extender (`/filter`, `/prioritize`) using GPU count, free VRAM, active requests and telemetry freshness. |
| `telemetry/` | Python | Node agent that publishes the GPU annotations the extender reads. |
| `worker/engine.py` | Python | Dependency-free synthetic backend (not a trained model), used for container and cluster tests. |

## Run locally

Requires Python 3.9+, Go 1.22+, and the packages in `requirements.txt`.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_model.py          # SmolLM2-135M-Instruct into models/
.venv/bin/python -m worker.server --port 8001       # real model, CPU, continuous batching
.venv/bin/python -m worker.server --port 8002
go run ./gateway --workers http://127.0.0.1:8001,http://127.0.0.1:8002 --listen :8000
```

```sh
curl -sS localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"What is a GPU? One sentence.","max_tokens":30}'
curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello","max_tokens":8,"stream":true}'
```

Open `http://localhost:8000/` for the dashboard. Worker options: `--policy continuous|static`, `--max-batch`, `--blocks`, `--block-size`, `--device cuda --attention cuda` on an NVIDIA GPU, and `--backend reference` for the synthetic model.

## Design

**Paged KV cache.** Each worker allocates one tensor per layer of shape `[blocks, block_size, kv_heads, head_dim]`. A request owns a logical block table that maps token positions to physical blocks, so sequences of different lengths share one pool without fragmentation. Prefill writes the prompt's keys and values through the table. Decode writes one slot per sequence and attends by gathering through the table, using either a vectorized PyTorch path or the CUDA kernel. Freed blocks are reused without being cleared, because recorded lengths and attention masks bound every read. A test covers this.

**Admission and batching.** At every engine tick the worker:
1. drops cancelled requests,
2. admits waiting requests up to `max_batch`, a per-tick prefill token budget, and available KV blocks,
3. runs one batched prefill for the new requests,
4. runs one batched decode for all active requests.

Under `continuous`, new requests join while others are still decoding. Under `static`, a batch must finish completely before the next is admitted. Admission reserves blocks for prompt plus `max_tokens` up front, so a request can never run out of cache mid-generation. A full queue returns 503.

**Gateway.** Health is probed in the background (every second by default) instead of per request. Requests go to the healthy worker with the fewest in-flight requests. If a worker refuses the connection, drops it before sending any response bytes, or returns 503 (at capacity), the gateway resends the buffered request to another worker. Streams are copied and flushed chunk by chunk.

**Scheduler extender.** A GPU pod passes `/filter` on a node only if all of these hold:
- the node has enough allocatable GPUs (counting init containers and sidecars),
- its telemetry timestamp is less than 30 s old,
- its free VRAM covers the pod's `inference.example.com/vram-mib` annotation.

`/prioritize` favors spare VRAM and fewer active requests.

## Results

All numbers below were measured on an Apple M5 (10 cores, 16 GB), CPU only, with 2 PyTorch threads, torch 2.6.0, SmolLM2-135M-Instruct, FP32, and the PyTorch attention path.

### Continuous vs. static batching

`benchmarks/compare.py` replays the same seeded Poisson arrival trace against both policies on the same model and code path:
- 24 requests at 20 req/s, output lengths drawn from {8, 16, 32, 64}, EOS ignored so lengths are fixed.
- Latency is measured from each request's scheduled arrival time, which avoids coordinated omission.
- Each repeat runs both policies, in alternating order.

Raw data: `benchmarks/results/`.

| max batch 4, 6 paired repeats | static | continuous | change |
|---|---|---|---|
| Throughput (tokens/s), mean ± sd | 133.0 ± 3.0 | 174.1 ± 2.9 | **+30.8%** (per-repeat range 28.8–33.9%) |
| P99 latency (ms), mean ± sd | 3293 ± 98 | 2378 ± 54 | **−27.7%** (per-repeat range 26.2–29.8%) |
| P99 time to first token (ms) | 2863 ± 88 | 1865 ± 55 | −34.8% |

With max batch 8, 48 requests and 4 repeats, the median change was +36.8% throughput and −35.0% P99. The spread was much wider (+7.6% to +46.4%) because another CPU-heavy job was running at the same time, so treat the batch-4 figures as the reliable ones.

### End to end through the gateway

Two real-model workers behind the gateway, run with `benchmarks/load.py`:

| Scenario | Succeeded | Throughput | P50 / P99 |
|---|---|---|---|
| 64 requests, concurrency 8, 32 tokens | 64 / 64 | 252 tokens/s | 1003 / 1095 ms |
| 96 requests; one worker killed with `kill -9` 3 s in | **96 / 96** | 302 tokens/s | 744 / 1633 ms |

In the failover run, the 4 requests that were in flight on the killed worker were retried on the surviving worker. When the worker restarted, the health loop returned it to rotation and it served traffic again.

### Kubernetes

`scripts/validate_kubernetes.py` ran against a 3-node kind cluster (Kubernetes v1.37.0) on Colima, with a second kube-scheduler that uses the extender. GPU inventory is **simulated** (extended resource plus annotations), because this Mac has no NVIDIA GPU. Results (`benchmarks/results/kubernetes-validation.json`):
- A pod asking for 12 GB of VRAM was placed on the node advertising 16 GB rather than the 4 GB node.
- A pod asking for more VRAM than any node has stayed Pending with "insufficient free GPU memory".
- A completion request sent through the in-cluster gateway succeeded.

The in-cluster workers run the synthetic reference backend.

### GPU and CUDA

The kernel's correctness test (`tests/test_llama.py`) compares it against the PyTorch path with fragmented page tables, in both FP32 and FP16. It needs an NVIDIA GPU, so it skips here. `artifacts/GPU-validation.ipynb` with `artifacts/inference-gpu-source.zip` runs the tests, model validation and the batching benchmark on a Colab GPU. **No GPU results have been recorded yet.** All throughput figures above are CPU figures.

## Tests

```sh
go vet ./... && go test -race ./...
.venv/bin/python -m unittest discover -s tests -v
```

- **Model:** paged prefill and decode match the Hugging Face reference logits across page boundaries for ragged batches; reallocated pages don't leak data from earlier requests.
- **Engine:** continuous and static admission, cancellation, and block release.
- **Detokenizer:** incremental output matches a full decode, including multibyte characters split across tokens.
- **Gateway:** routing, skipping unhealthy workers, failover with the request body resent, retry on 503, incremental streaming, and health recovery.
- **Scheduler:** filter and prioritize behavior.

## Limitations

- **Distribution is at the replica level.** Each worker holds the whole model. There is no tensor or pipeline parallelism, and nothing like prefix caching or KV transfer between workers.
- **KV blocks are reserved up front** for the whole prompt plus `max_tokens`. This is simpler than vLLM-style on-demand allocation with preemption, but less of the cache is in use at any moment.
- A stream that fails **after** tokens have been sent cannot be retried. The client sees the stream end without a `done` event.
- Decoding is greedy. Only Llama-architecture checkpoints with unscaled RoPE are supported.
- The extender relies on a telemetry publisher to keep its node annotations current, and it requires control of the scheduler configuration, which many managed Kubernetes services don't allow.
