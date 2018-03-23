/*      This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "../libethcore/MinerCommon.h"
#include "ethash_cuda_miner_kernel.h"
#include "ethash_cuda_miner_kernel_globals.h"
#include "cuda_helper.h"
#include "fnv.cuh"
#include <stdio.h>

#define copy(dst, src, count) for (int i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }

#include "keccak.cuh"
#include "dagger_shuffled.cuh"

template <uint32_t _PARALLEL_HASH>
__global__ void
ethash_search(
    volatile search_results* g_output,
    uint64_t start_nonce
)
{
	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint2 mix[4];
	if (!compute_hash<_PARALLEL_HASH>(start_nonce + gid, d_target, mix))
		return;
	uint32_t i = atomicInc((unsigned*)&g_output->count, 0xffffffff);
	if (i)
		return;
	g_output->gid = gid;
	g_output->mix[0] = devectorize(mix[0]);
	g_output->mix[1] = devectorize(mix[1]);
	g_output->mix[2] = devectorize(mix[2]);
	g_output->mix[3] = devectorize(mix[3]);
}

void run_ethash_search(
    uint32_t blocks,
    uint32_t threads,
    cudaStream_t stream,
    volatile search_results* g_output,
    uint64_t start_nonce,
    uint32_t parallelHash
)
{
	switch (parallelHash) {
	case 1: ethash_search <1> <<<blocks, threads, 0, stream>>>(g_output, start_nonce); break;
	case 2: ethash_search <2> <<<blocks, threads, 0, stream>>>(g_output, start_nonce); break;
	case 4: ethash_search <4> <<<blocks, threads, 0, stream>>>(g_output, start_nonce); break;
	case 8: ethash_search <8> <<<blocks, threads, 0, stream>>>(g_output, start_nonce); break;
	default: ethash_search <4> <<<blocks, threads, 0, stream>>>(g_output, start_nonce); break;
	}
	CUDA_SAFE_CALL(cudaGetLastError());
}

#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS (64/4)

#define shuffl4(_a, _b) __shfl_sync(0xFFFFFFFF, _a, _b, 4)

__global__ void
ethash_calculate_dag_item(uint32_t start)
{
	uint32_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
	if (node_index > d_dag_size * 2) return;

	hash200_t dag_node;
	copy(dag_node.uint4s, d_light[node_index % d_light_size].uint4s, 4);
	dag_node.words[0] ^= node_index;
	SHA3_512(dag_node.uint2s);

	const int thread_id = threadIdx.x & 3;

	for (uint32_t i = 0; i != ETHASH_DATASET_PARENTS; ++i) {
		uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % d_light_size;
		for (uint32_t t = 0; t < 4; t++) {

			uint32_t shuffle_index = shuffl4(parent_index, t);

			uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
			for (int w = 0; w < 4; w++) {
				uint4 s4 = make_uint4(shuffl4(p4.x, w), shuffl4(p4.y, w), shuffl4(p4.z, w), shuffl4(p4.w, w));
				if (t == thread_id)
					dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
			}
		}
	}
	SHA3_512(dag_node.uint2s);
	hash64_t* dag_nodes = (hash64_t*)d_dag;

	for (uint32_t t = 0; t < 4; t++) {
		uint32_t shuffle_index = shuffl4(node_index, t);
		uint4 s[4];
		for (uint32_t w = 0; w < 4; w++)
			s[w] = make_uint4(shuffl4(dag_node.uint4s[w].x, t), shuffl4(dag_node.uint4s[w].y, t), shuffl4(dag_node.uint4s[w].z, t),
			                  shuffl4(dag_node.uint4s[w].w, t));
		dag_nodes[shuffle_index].uint4s[thread_id] = s[thread_id];
	}
}

void ethash_generate_dag(
    uint64_t dag_size,
    uint32_t blocks,
    uint32_t threads,
    cudaStream_t stream,
    int device
)
{
	uint32_t const work = (uint32_t)(dag_size / sizeof(hash64_t));

	uint32_t fullRuns = work / (blocks * threads);
	uint32_t const restWork = work % (blocks * threads);
	if (restWork > 0) fullRuns++;
	for (uint32_t i = 0; i < fullRuns; i++) {
		ethash_calculate_dag_item <<<blocks, threads, 0, stream>>>(i * blocks * threads);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}
	CUDA_SAFE_CALL(cudaGetLastError());
}

void set_constants(
    hash128_t* _dag,
    uint32_t _dag_size,
    hash64_t* _light,
    uint32_t _light_size
)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag, &_dag, sizeof(hash128_t*)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag_size, &_dag_size, sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_light, &_light, sizeof(hash64_t*)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_light_size, &_light_size, sizeof(uint32_t)));
}

void set_header_and_target(
    hash32_t _header,
    uint64_t _target
)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_header, &_header, sizeof(hash32_t)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_target, &_target, sizeof(uint64_t)));
}

