"""Reproducible arrival-trace comparison; same model and execution in both modes.

Fixed output lengths (EOS ignored) isolate scheduling from model stopping rules.
Latency includes waiting since the scheduled arrival, avoiding coordinated omission.
"""
import argparse
import json
import math
import platform
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from worker.llama import load_pretrained
from worker.batched import BatchedEngine


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def run(model, trace, policy, max_batch):
    engine = BatchedEngine(model, max_batch=max_batch, policy=policy)
    pending, finished, index = {}, {}, 0
    start = time.monotonic()
    while len(finished) < len(trace):
        now = time.monotonic() - start
        while index < len(trace) and trace[index]['arrival_s'] <= now:
            item = trace[index]
            request = engine.submit(item['prompt'], item['max_tokens'], ignore_eos=True)
            request.submitted_at = start + item['arrival_s']
            pending[request.id] = request
            index += 1
        engine.tick()
        for request_id, request in list(pending.items()):
            while not request.events.empty():
                event = request.events.get_nowait()
                if 'error' in event:
                    raise RuntimeError(event['error'])
                if event.get('done'):
                    finished[request_id] = {
                        'latency_s': time.monotonic() - request.submitted_at,
                        'ttft_s': request.first_token_at - request.submitted_at,
                        'tokens': request.produced,
                    }
                    del pending[request_id]
        if not engine.active and index < len(trace):
            time.sleep(min(.001, max(0, trace[index]['arrival_s'] - (time.monotonic() - start))))
    elapsed = time.monotonic() - start
    assert len(model.cache.free) == model.cache.total_blocks
    latencies = [r['latency_s'] for r in finished.values()]
    return {'policy': policy, 'seconds': elapsed,
            'tokens_per_second': sum(r['tokens'] for r in finished.values()) / elapsed,
            'requests_per_second': len(trace) / elapsed,
            'p50_ms': statistics.median(latencies) * 1000,
            'p99_ms': percentile(latencies, .99) * 1000,
            'ttft_p99_ms': percentile([r['ttft_s'] for r in finished.values()], .99) * 1000,
            'max_observed_batch': engine.max_observed_batch,
            'requests': list(finished.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/smollm2-135m-instruct')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--attention', default='torch')
    parser.add_argument('--requests', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output', default='.run/batching-comparison.json')
    args = parser.parse_args()
    if min(args.requests, args.repeats, args.batch) < 1:
        parser.error('requests, repeats and batch must be positive')
    torch.set_num_threads(2)
    randomizer = random.Random(42)
    trace, arrival = [], 0.0
    for _ in range(args.requests):
        trace.append({'arrival_s': arrival, 'prompt': 'Explain what a computer does in simple words.',
                      'max_tokens': randomizer.choice([8, 16, 32, 64])})
        arrival += randomizer.expovariate(20)
    model = load_pretrained(args.model, args.device, blocks=256, attention=args.attention)
    run(model, [{'arrival_s': 0, 'prompt': 'Hello', 'max_tokens': 8}], 'continuous', args.batch)
    trials = []
    for repeat in range(args.repeats):
        for policy in (('static', 'continuous') if repeat % 2 == 0 else ('continuous', 'static')):
            result = run(model, trace, policy, args.batch)
            result['repeat'] = repeat
            trials.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != 'requests'}), flush=True)
    def spread(values):
        return {'median': statistics.median(values), 'mean': statistics.mean(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values), 'max': max(values)}

    summary = {}
    for policy in ('static', 'continuous'):
        selected = [r for r in trials if r['policy'] == policy]
        summary[policy] = {metric: statistics.median(r[metric] for r in selected)
                           for metric in ('tokens_per_second', 'p99_ms', 'ttft_p99_ms')}
        summary[policy]['spread'] = {metric: spread([r[metric] for r in selected])
                                     for metric in ('tokens_per_second', 'p99_ms', 'ttft_p99_ms')}
    summary['throughput_change_pct'] = 100 * (summary['continuous']['tokens_per_second'] / summary['static']['tokens_per_second'] - 1)
    summary['p99_reduction_pct'] = 100 * (1 - summary['continuous']['p99_ms'] / summary['static']['p99_ms'])
    # Paired per-repeat changes show whether the effect holds in every trial, not just the median.
    paired = [(next(r for r in trials if r['repeat'] == i and r['policy'] == 'static'),
               next(r for r in trials if r['repeat'] == i and r['policy'] == 'continuous'))
              for i in range(args.repeats)]
    summary['per_repeat_throughput_change_pct'] = spread(
        [100 * (c['tokens_per_second'] / s['tokens_per_second'] - 1) for s, c in paired])
    summary['per_repeat_p99_reduction_pct'] = spread([100 * (1 - c['p99_ms'] / s['p99_ms']) for s, c in paired])
    report = {'model': args.model, 'machine': platform.platform(), 'processor': platform.machine(),
              'torch_threads': torch.get_num_threads(), 'requests': args.requests, 'repeats': args.repeats, 'device': str(model.device), 'attention': args.attention,
              'torch_version': torch.__version__, 'seed': 42, 'max_batch': args.batch,
              'fixed_output_lengths': True, 'trace': trace, 'trials': trials, 'summary': summary}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
