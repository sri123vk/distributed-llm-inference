// One CTA per sequence/query head. Threads score tokens in parallel and reduce
// the softmax in FP32. KV heads are shared by groups of query heads (GQA).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void attention_kernel(
    const scalar_t* q, const scalar_t* keys, const scalar_t* values,
    const int* tables, const int* lengths, scalar_t* output,
    int table_width, int block_size, int heads, int kv_heads, int dim) {
  int seq = blockIdx.x, head = blockIdx.y, thread = threadIdx.x;
  int kv_head = head / (heads / kv_heads), length = lengths[seq];
  extern __shared__ float scores[];
  __shared__ float reduction[128];
  float maximum = -INFINITY;
  for (int t = thread; t < length; t += blockDim.x) {
    int page = tables[seq * table_width + t / block_size];
    long long base = ((static_cast<long long>(page) * block_size + t % block_size) * kv_heads + kv_head) * dim;
    float score = 0.f;
    for (int d = 0; d < dim; ++d)
      score += static_cast<float>(q[(seq * heads + head) * dim + d]) * static_cast<float>(keys[base + d]);
    score *= rsqrtf(static_cast<float>(dim));
    scores[t] = score;
    maximum = fmaxf(maximum, score);
  }
  reduction[thread] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (thread < stride) reduction[thread] = fmaxf(reduction[thread], reduction[thread + stride]);
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float sum = 0.f;
  for (int t = thread; t < length; t += blockDim.x) {
    scores[t] = expf(scores[t] - maximum);
    sum += scores[t];
  }
  reduction[thread] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (thread < stride) reduction[thread] += reduction[thread + stride];
    __syncthreads();
  }
  for (int d = thread; d < dim; d += blockDim.x) {
    float value = 0.f;
    for (int t = 0; t < length; ++t) {
      int page = tables[seq * table_width + t / block_size];
      long long index = ((static_cast<long long>(page) * block_size + t % block_size) * kv_heads + kv_head) * dim + d;
      value += scores[t] * static_cast<float>(values[index]);
    }
    output[(seq * heads + head) * dim + d] = static_cast<scalar_t>(length ? value / reduction[0] : 0.f);
  }
}

torch::Tensor paged_attention_cuda(torch::Tensor q, torch::Tensor keys, torch::Tensor values,
                                  torch::Tensor tables, torch::Tensor lengths, int64_t max_length) {
  const c10::cuda::CUDAGuard guard(q.device());
  auto output = torch::empty_like(q);
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "paged_attention_cuda", [&] {
    attention_kernel<scalar_t><<<dim3(q.size(0), q.size(1)), 128,
                               max_length * sizeof(float), stream.stream()>>>(
      q.data_ptr<scalar_t>(), keys.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
      tables.data_ptr<int>(), lengths.data_ptr<int>(), output.data_ptr<scalar_t>(),
      tables.size(1), keys.size(1), q.size(1), keys.size(2), q.size(2));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
