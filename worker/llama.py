"""Batched Llama inference with an owned, physical-page KV cache.

Hugging Face supplies checkpoint loading and layer weights. Prefill and decode
are executed here, including position encoding and cache access.
"""
from pathlib import Path
import math

import torch
import torch.nn.functional as F


class TensorPagedKV:
    def __init__(self, config, blocks=256, block_size=16, device='cpu', dtype=torch.float32):
        self.block_size = block_size
        self.total_blocks = blocks
        self.free = list(range(blocks - 1, -1, -1))
        self.tables = {}
        self.lengths = {}
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        shape = (config.num_hidden_layers, blocks, block_size, config.num_key_value_heads, head_dim)
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros_like(self.keys)

    def reserve(self, request_id, tokens):
        if request_id in self.tables:
            raise ValueError('duplicate request ID')
        count = math.ceil(tokens / self.block_size)
        if count > len(self.free):
            return False
        self.tables[request_id] = [self.free.pop() for _ in range(count)]
        self.lengths[request_id] = 0
        return True

    def release(self, request_id):
        # Stale cells are never read: lengths and attention masks bound all reads.
        self.free.extend(self.tables.pop(request_id, []))
        self.lengths.pop(request_id, None)

    def slots(self, ids, positions):
        physical, offsets = [], []
        for request_id, sequence_positions in zip(ids, positions):
            for position in sequence_positions:
                physical.append(self.tables[request_id][position // self.block_size])
                offsets.append(position % self.block_size)
        return (torch.tensor(physical, device=self.keys.device, dtype=torch.long),
                torch.tensor(offsets, device=self.keys.device, dtype=torch.long))

    def metadata(self, ids):
        lengths = [self.lengths[i] for i in ids]
        max_length = max(lengths)
        width = math.ceil(max_length / self.block_size)
        tables = [self.tables[i][:width] + [0] * max(0, width - len(self.tables[i])) for i in ids]
        return (torch.tensor(tables, device=self.keys.device, dtype=torch.int32),
                torch.tensor(lengths, device=self.keys.device, dtype=torch.int32), max_length)


def torch_paged_attention(query, keys, values, tables, lengths, max_length):
    """Vectorized fallback: gather pages, then apply masked grouped-query attention."""
    block_size = keys.shape[1]
    positions = torch.arange(max_length, device=query.device)
    physical = tables.long()[:, positions // block_size]
    offsets = positions % block_size
    k = keys[physical, offsets].transpose(1, 2)
    v = values[physical, offsets].transpose(1, 2)
    groups = query.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    mask = positions[None, :] < lengths[:, None]
    return F.scaled_dot_product_attention(query.unsqueeze(2), k, v,
                                         attn_mask=mask[:, None, None, :]).squeeze(2)


_cuda_module = None


def cuda_paged_attention(query, keys, values, tables, lengths, max_length):
    global _cuda_module
    if _cuda_module is None:
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parents[1] / 'cuda'
        _cuda_module = load(name='inference_paged_attention_v2',
                            sources=[str(root / 'binding.cpp'), str(root / 'paged_attention.cu')],
                            extra_cuda_cflags=['-O3'], extra_cflags=['-O3'])
    return _cuda_module.paged_attention(query.contiguous(), keys, values,
                                        tables.contiguous(), lengths.contiguous(), max_length)


class PagedLlama:
    def __init__(self, model, tokenizer=None, blocks=256, block_size=16, attention='torch'):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = model.config
        if self.config.model_type != 'llama' or self.config.rope_scaling is not None:
            raise ValueError('Only Llama checkpoints with unscaled RoPE are supported')
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.heads = self.config.num_attention_heads
        self.kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, 'head_dim', self.config.hidden_size // self.heads)
        self.cache = TensorPagedKV(self.config, blocks, block_size, self.device, self.dtype)
        self.attention = attention
        if attention == 'cuda' and self.device.type != 'cuda':
            raise ValueError('--attention cuda requires an NVIDIA CUDA device')
        self.inv_freq = 1.0 / (self.config.rope_theta ** (
            torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32) / self.head_dim))

    def rope(self, q, k, positions):
        frequencies = positions.float()[..., None] * self.inv_freq
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = angles.cos().to(q.dtype).unsqueeze(2), angles.sin().to(q.dtype).unsqueeze(2)

        def rotate(tensor):
            a, b = tensor.chunk(2, dim=-1)
            return torch.cat((-b, a), dim=-1)

        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin

    @torch.inference_mode()
    def prefill(self, ids, token_lists):
        batch, width = len(ids), max(map(len, token_lists))
        tokens = torch.zeros((batch, width), device=self.device, dtype=torch.long)
        valid = torch.zeros((batch, width), device=self.device, dtype=torch.bool)
        for row, (request_id, sequence) in enumerate(zip(ids, token_lists)):
            tokens[row, :len(sequence)] = torch.tensor(sequence, device=self.device)
            valid[row, :len(sequence)] = True
            self.cache.lengths[request_id] = len(sequence)
        slots = self.cache.slots(ids, [range(len(s)) for s in token_lists])
        positions = torch.arange(width, device=self.device)[None, :].expand(batch, -1)
        x = self.model.model.embed_tokens(tokens)
        for index, layer in enumerate(self.model.model.layers):
            residual = x
            normalized = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(normalized).view(batch, width, self.heads, self.head_dim)
            k = attn.k_proj(normalized).view(batch, width, self.kv_heads, self.head_dim)
            v = attn.v_proj(normalized).view(batch, width, self.kv_heads, self.head_dim)
            q, k = self.rope(q, k, positions)
            self.cache.keys[index][slots] = k[valid]
            self.cache.values[index][slots] = v[valid]
            groups = self.heads // self.kv_heads
            context = F.scaled_dot_product_attention(q.transpose(1, 2),
                k.repeat_interleave(groups, dim=2).transpose(1, 2),
                v.repeat_interleave(groups, dim=2).transpose(1, 2), is_causal=True)
            x = residual + attn.o_proj(context.transpose(1, 2).reshape(batch, width, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        last = torch.tensor([len(s) - 1 for s in token_lists], device=self.device)
        hidden = self.model.model.norm(x[torch.arange(batch, device=self.device), last])
        return self.model.lm_head(hidden).float()

    @torch.inference_mode()
    def decode(self, ids, tokens):
        batch = len(ids)
        positions_list = [self.cache.lengths[i] for i in ids]
        slots = self.cache.slots(ids, [[p] for p in positions_list])
        positions = torch.tensor(positions_list, device=self.device)[:, None]
        for request_id in ids:
            self.cache.lengths[request_id] += 1
        tables, lengths, max_length = self.cache.metadata(ids)
        token_tensor = torch.tensor(tokens, device=self.device, dtype=torch.long)[:, None]
        x = self.model.model.embed_tokens(token_tensor)
        attention = cuda_paged_attention if self.attention == 'cuda' else torch_paged_attention
        for index, layer in enumerate(self.model.model.layers):
            residual = x
            normalized = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(normalized).view(batch, 1, self.heads, self.head_dim)
            k = attn.k_proj(normalized).view(batch, 1, self.kv_heads, self.head_dim)
            v = attn.v_proj(normalized).view(batch, 1, self.kv_heads, self.head_dim)
            q, k = self.rope(q, k, positions)
            self.cache.keys[index][slots] = k[:, 0]
            self.cache.values[index][slots] = v[:, 0]
            context = attention(q[:, 0], self.cache.keys[index], self.cache.values[index],
                                tables, lengths, max_length)
            x = residual + attn.o_proj(context.reshape(batch, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.model.lm_head(self.model.model.norm(x[:, 0])).float()


def load_pretrained(model_path, device='cpu', blocks=256, block_size=16, attention='torch'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
        use_safetensors=True, trust_remote_code=False, attn_implementation='eager').to(device)
    return PagedLlama(model, tokenizer, blocks, block_size, attention)
