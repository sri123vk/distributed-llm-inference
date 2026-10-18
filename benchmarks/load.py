"""End-to-end throughput and tail-latency probe for /v1/completions."""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    payload = json.dumps({"prompt": "Benchmark prompt ", "max_tokens": args.max_tokens}).encode()

    def one(_):
        request = urllib.request.Request(args.url, payload, {"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
        except (OSError, ValueError) as exc:  # HTTP errors, resets and bad bodies count as failures
            return time.perf_counter() - start, None, str(exc)
        return time.perf_counter() - start, result["tokens"], None

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.requests)))
    elapsed = time.perf_counter() - start
    failures = [r[2] for r in results if r[2] is not None]
    results = [r for r in results if r[2] is None]
    if not results:
        raise SystemExit("all requests failed: " + failures[0])
    latency = sorted(r[0] for r in results)
    p99 = latency[min(len(latency) - 1, int(0.99 * len(latency)))]
    print(json.dumps({
        "requests": len(results) + len(failures), "succeeded": len(results), "failed": len(failures),
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed, 3),
        "requests_per_second": round(len(results) / elapsed, 2),
        "tokens_per_second": round(sum(r[1] for r in results) / elapsed, 2),
        "latency_p50_ms": round(statistics.median(latency) * 1000, 2),
        "latency_p99_ms": round(p99 * 1000, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
