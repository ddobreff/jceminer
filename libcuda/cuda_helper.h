/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Input:       77665544 33221100
// Output:      00112233 44556677
__device__ __forceinline__ uint64_t cuda_swab64(const uint64_t x)
{
    uint64_t result;
    uint2 t;
    asm("mov.b64 {%0,%1},%2; \n\t"
        : "=r"(t.x), "=r"(t.y) : "l"(x));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);
    asm("mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}

__device__ __forceinline__ uint64_t devectorize(uint2 x)
{
    uint64_t result;
    asm("mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result) : "r"(x.x), "r"(x.y));
    return result;
}

__device__ __forceinline__ uint2 vectorize(const uint64_t x)
{
    uint2 result;
    asm("mov.b64 {%0,%1},%2; \n\t"
        : "=r"(result.x), "=r"(result.y) : "l"(x));
    return result;
}

__device__ __forceinline__ void devectorize2(uint4 inn, uint2& x, uint2& y)
{
    x.x = inn.x;
    x.y = inn.y;
    y.x = inn.z;
    y.y = inn.w;
}

__device__ __forceinline__ uint4 vectorize2(uint2 x, uint2 y)
{
    uint4 result;
    result.x = x.x;
    result.y = x.y;
    result.z = y.x;
    result.w = y.y;

    return result;
}

__device__ __forceinline__ uint4 vectorize2(uint2 x)
{
    uint4 result;
    result.x = x.x;
    result.y = x.y;
    result.z = x.x;
    result.w = x.y;
    return result;
}

static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

static __device__ __forceinline__ uint2 operator& (uint2 a, uint2 b)
{
    return make_uint2(a.x & b.x, a.y & b.y);
}

static __device__ __forceinline__ uint2 operator~(uint2 a)
{
    return make_uint2(~a.x, ~a.y);
}


static __device__ __forceinline__ void operator^= (uint2& a, uint2 b)
{
    a = a ^ b;
}

static __forceinline__ __device__ uint2 ROL2(const uint2 a, const int offset)
{
    uint2 result;
    if (offset >= 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
    }
    else {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    }
    return result;
}

static __device__ __forceinline__
uint32_t bfe(uint32_t x, uint32_t bit, uint32_t numBits)
{
    uint32_t ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
    return ret;

}

