#include <torch/extension.h>

torch::Tensor paged_attention_cuda(torch::Tensor, torch::Tensor, torch::Tensor,
                                  torch::Tensor, torch::Tensor, int64_t);

torch::Tensor paged_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             torch::Tensor tables, torch::Tensor lengths, int64_t max_length) {
  TORCH_CHECK(q.is_cuda(), "query must be on CUDA");
  for (const auto& tensor : {q, k, v, tables, lengths}) {
    TORCH_CHECK(tensor.device() == q.device(), "all tensors must be on the same GPU");
    TORCH_CHECK(tensor.is_contiguous(), "all tensors must be contiguous");
  }
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && k.sizes() == v.sizes(), "invalid Q/K/V shapes");
  TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && q.size(2) <= 256, "invalid query shape");
  TORCH_CHECK(k.size(0) > 0 && k.size(1) > 0 && k.size(2) > 0 && k.size(3) == q.size(2), "invalid cache shape");
  TORCH_CHECK(q.size(1) % k.size(2) == 0, "query heads must be a multiple of KV heads");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16, "use FP32 or FP16");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(), "dtype mismatch");
  TORCH_CHECK(tables.scalar_type() == torch::kInt32 && lengths.scalar_type() == torch::kInt32, "metadata must be int32");
  TORCH_CHECK(tables.dim() == 2 && tables.size(0) == q.size(0), "invalid block tables");
  TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == q.size(0), "invalid lengths");
  TORCH_CHECK(max_length > 0 && max_length <= 8192 && tables.size(1) * k.size(1) >= max_length, "invalid context bound");
  // Metadata comes from the owned Python allocator: valid physical pages and
  // 0 <= lengths[i] <= max_length are required.
  return paged_attention_cuda(q, k, v, tables, lengths, max_length);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("paged_attention", &paged_attention, "Paged GQA attention (CUDA)");
}
