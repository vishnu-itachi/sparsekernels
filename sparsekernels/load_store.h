#ifndef THIRD_PARTY_SPARSEKERNEL_LOAD_STORE_H_
#define THIRD_PARTY_SPARSEKERNEL_LOAD_STORE_H_

/**
 * @file @brief Defines utilities for loading and storing data.
 */

#include "sparsekernel/cuda_utils.h"
#include <cstring>

namespace sparsekernel {

template <class To, class From>
__device__ __forceinline__ To BitCast(const From &src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
__device__ __forceinline__ void Store(const T &value, T *ptr) {
  *ptr = value;
}

__device__ __forceinline__ void Store(const half8 &value, half8 *ptr) {
  *reinterpret_cast<float4 *>(ptr) = BitCast<float4>(value);
}

__device__ __forceinline__ void Store(const half4 &value, half4 *ptr) {
  *reinterpret_cast<float2 *>(ptr) = BitCast<float2>(value);
}

__device__ __forceinline__ void Store(const short8 &value, short8 *ptr) {
  *reinterpret_cast<int4 *>(ptr) = BitCast<int4>(value);
}

__device__ __forceinline__ void Store(const short4 &value, short4 *ptr) {
  *reinterpret_cast<int2 *>(ptr) = BitCast<int2>(value);
}

template <typename T> __device__ __forceinline__ T Load(const T *address) {
  return __ldg(address);
}

__device__ __forceinline__ half4 Load(const half4 *address) {
  float2 x = __ldg(reinterpret_cast<const float2 *>(address));
  return BitCast<half4>(x);
}

__device__ __forceinline__ half8 Load(const half8 *address) {
  float4 x = __ldg(reinterpret_cast<const float4 *>(address));
  return BitCast<half8>(x);
}

__device__ __forceinline__ short4 Load(const short4 *address) {
  int2 x = __ldg(reinterpret_cast<const int2 *>(address));
  return BitCast<short4>(x);
}

__device__ __forceinline__ short8 Load(const short8 *address) {
  int4 x = __ldg(reinterpret_cast<const int4 *>(address));
  return BitCast<short8>(x);
}

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_LOAD_STORE_H_
