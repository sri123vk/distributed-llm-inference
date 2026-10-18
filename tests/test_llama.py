import unittest
try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    raise unittest.SkipTest('Install requirements.txt to test the trained-model path')

from worker.llama import PagedLlama, torch_paged_attention, cuda_paged_attention
from pathlib import Path

from worker.batched import BatchedEngine, detokenize


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [3 + ord(c) % 60 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]['content'])

    def decode(self, ids, **kwargs):
        return ''.join(chr(65 + i % 26) for i in ids)


def make_model(blocks=32):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    return PagedLlama(LlamaForCausalLM(config).eval(), Tokenizer(), blocks=blocks, block_size=4)


class ModelCorrectness(unittest.TestCase):
    @torch.inference_mode()
    def test_ragged_batch_matches_huggingface_across_page_boundaries(self):
        model = make_model()
        sequences = [[1, 5, 9], [1, 3, 4, 7, 11]]
        ids = ['a', 'b']
        for request_id, sequence in zip(ids, sequences):
            self.assertTrue(model.cache.reserve(request_id, len(sequence) + 12))
        logits = model.prefill(ids, sequences)
        for _ in range(10):
            for row, sequence in enumerate(sequences):
                reference = model.model(torch.tensor([sequence]), use_cache=False).logits[0, -1]
                torch.testing.assert_close(logits[row], reference, atol=2e-5, rtol=2e-5)
            next_tokens = logits.argmax(-1).tolist()
            for sequence, token in zip(sequences, next_tokens):
                sequence.append(token)
            logits = model.decode(ids, next_tokens)
        for request_id in ids:
            model.cache.release(request_id)
        self.assertEqual(len(model.cache.free), model.cache.total_blocks)

    @torch.inference_mode()
    def test_reallocated_pages_do_not_leak_previous_request(self):
        model = make_model()
        model.cache.reserve('old', 12)
        model.prefill(['old'], [[1, 6, 7, 8, 9]])
        model.cache.release('old')
        model.cache.reserve('new', 12)
        model.prefill(['new'], [[1, 42]])
        result = model.decode(['new'], [8])[0]
        expected = model.model(torch.tensor([[1, 42, 8]]), use_cache=False).logits[0, -1]
        torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)

    def test_continuous_admission_static_policy_and_cancellation(self):
        for policy in ('continuous', 'static'):
            engine = BatchedEngine(make_model(), max_batch=2, policy=policy, max_context=128)
            first = engine.submit('hello', 4, ignore_eos=True)
            engine.tick()
            second = engine.submit('world', 3, ignore_eos=True)
            engine.tick()
            self.assertEqual(second.produced, 1 if policy == 'continuous' else 0)
            first.cancelled = True
            while engine.active or engine.waiting:
                engine.tick()
            self.assertEqual(second.produced, 3)
            self.assertEqual(len(engine.kv.free), engine.kv.total_blocks)


class ByteTokenizer:
    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8', errors='replace')


def stream(tokenizer, ids):
    pieces, prefix, read = [], 0, 0
    for n in range(1, len(ids) + 1):
        piece, prefix, read = detokenize(tokenizer, ids[:n], prefix, read, final=n == len(ids))
        pieces.append(piece)
    return pieces


class IncrementalDetokenization(unittest.TestCase):
    def test_multibyte_characters_split_across_tokens(self):
        ids = list('héllo → 世界 ok'.encode())
        pieces = stream(ByteTokenizer(), ids)
        self.assertEqual(''.join(pieces), 'héllo → 世界 ok')
        self.assertNotIn('\ufffd', ''.join(pieces))

    def test_truncated_character_is_flushed_on_final_token(self):
        ids = list('a世'.encode())[:-1]
        self.assertEqual(''.join(stream(ByteTokenizer(), ids)), 'a\ufffd')

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'models/smollm2-135m-instruct').is_dir(),
                         'SmolLM2 checkpoint not downloaded')
    def test_matches_full_decode_with_real_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(__file__).resolve().parents[1] / 'models/smollm2-135m-instruct'))
        text = 'Paged attention — 页面 cache: naïve “quotes”, emoji 🚀 and  double  spaces.\nDone.'
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(''.join(stream(tokenizer, ids)), expected)


@unittest.skipUnless(torch.cuda.is_available(), 'NVIDIA GPU required')
class CUDAKernelCorrectness(unittest.TestCase):
    def test_cuda_paged_gqa_matches_torch_with_fragmented_pages(self):
        for dtype in (torch.float32, torch.float16):
            torch.manual_seed(11)
            q = torch.randn(3, 4, 32, device='cuda', dtype=dtype)
            keys = torch.randn(12, 16, 2, 32, device='cuda', dtype=dtype)
            values = torch.randn_like(keys)
            tables = torch.tensor([[9, 2, 5], [1, 8, 4], [7, 0, 6]], device='cuda', dtype=torch.int32)
            lengths = torch.tensor([1, 17, 35], device='cuda', dtype=torch.int32)
            expected = torch_paged_attention(q, keys, values, tables, lengths, 35)
            actual = cuda_paged_attention(q, keys, values, tables, lengths, 35)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
