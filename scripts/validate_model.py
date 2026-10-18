"""Validate checkpoint logits and actual generation through the paged engine."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from worker.llama import load_pretrained
from worker.batched import BatchedEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/smollm2-135m-instruct')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--attention', default='torch')
    parser.add_argument('--output', default='.run/trained-model-validation.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_pretrained(args.model, args.device, blocks=128, attention=args.attention)
    prompts = ['The capital of France is', 'A computer is a machine that']
    sequences = [model.tokenizer.encode(p) for p in prompts]
    ids = ['parity-a', 'parity-b']
    for request_id, sequence in zip(ids, sequences):
        model.cache.reserve(request_id, len(sequence) + 8)
    errors = []
    with torch.inference_mode():
        logits = model.prefill(ids, sequences)
        for _ in range(5):
            for index, sequence in enumerate(sequences):
                expected = model.model(torch.tensor([sequence], device=model.device), use_cache=False).logits[0, -1].float()
                errors.append((logits[index] - expected).abs().max().item())
                torch.testing.assert_close(logits[index], expected,
                    atol=.08 if args.device == 'cuda' else .0002,
                    rtol=.01 if args.device == 'cuda' else .0002)
            tokens = logits.argmax(-1).tolist()
            for sequence, token in zip(sequences, tokens):
                sequence.append(token)
            logits = model.decode(ids, tokens)
    for request_id in ids:
        model.cache.release(request_id)
    engine = BatchedEngine(model)
    start = time.perf_counter()
    request = engine.submit('What is the capital of France?', 48)
    while engine.active or engine.waiting:
        engine.tick()
    events = []
    while not request.events.empty():
        events.append(request.events.get_nowait())
    assert not any('error' in event for event in events), events
    assert len(model.cache.free) == model.cache.total_blocks
    report = {'model': args.model, 'device': str(model.device), 'attention': args.attention,
              'max_logit_error': max(errors), 'prompt': request.prompt,
              'output': ''.join(request.output), 'generated_tokens': request.produced,
              'seconds': time.perf_counter() - start, 'cache_released': True}
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
