/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cuda_runtime.h>

// It is virtually impossible to get more than
// one solution per stream hash calculation
typedef struct {
	uint32_t count;
	uint32_t gid;
	uint64_t mix[4];
} search_results;

#define ACCESSES 64
#define THREADS_PER_HASH (128 / 16)

typedef struct {
	uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;

typedef struct {
	uint4	 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef union {
	uint32_t words[64 / sizeof(uint32_t)];
	uint2	 uint2s[64 / sizeof(uint2)];
	uint4	 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union {
	uint32_t words[200 / sizeof(uint32_t)];
	uint2	 uint2s[200 / sizeof(uint2)];
	uint4	 uint4s[200 / sizeof(uint4)];
} hash200_t;

void set_constants(
    hash128_t* _dag,
    uint32_t _dag_size,
    hash64_t* _light,
    uint32_t _light_size
);

void set_header_and_target(
    hash32_t _header,
    uint64_t _target
);

void run_ethash_search(
    uint32_t search_batch_size,
    uint32_t workgroup_size,
    cudaStream_t stream,
    volatile search_results* g_output,
    uint64_t start_nonce,
    uint32_t parallelHash
);

void ethash_generate_dag(
    uint64_t dag_size,
    uint32_t blocks,
    uint32_t threads,
    cudaStream_t stream
);

struct cuda_runtime_error : public virtual std::runtime_error {
	cuda_runtime_error(std::string msg) : std::runtime_error(msg) {}
};

#define CUDA_SAFE_CALL(call)				\
do {							\
	cudaError_t err = call;				\
	if (cudaSuccess != err) {			\
		std::stringstream ss;			\
		ss << "CUDA error in func " 		\
			<< __FUNCTION__ 		\
			<< " at line "			\
			<< __LINE__			\
			<< ' '				\
			<< cudaGetErrorString(err);	\
		throw cuda_runtime_error(ss.str());	\
	}						\
} while (0)

