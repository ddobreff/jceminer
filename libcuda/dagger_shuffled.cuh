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

#include "ethash_cuda_miner_kernel_globals.h"
#include "ethash_cuda_miner_kernel.h"
#include "cuda_helper.h"

#define shuffle(_a, _b) __shfl_sync(0xFFFFFFFF, _a, _b, THREADS_PER_HASH);

template <uint32_t _PARALLEL_HASH>
__device__ __forceinline__ bool compute_hash(
    uint64_t nonce,
    uint64_t target,
    uint2* mix_hash)
{
	// sha3_512(header .. nonce)
	uint2 state[12];

	state[4] = vectorize(nonce);

	keccak_f1600_init(state);

	// Threads work together in this phase in groups of 8.
	const int thread_id  = threadIdx.x & (THREADS_PER_HASH - 1);
	const int mix_idx    = thread_id & 3;

	for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH) {
		uint4 mix[_PARALLEL_HASH];
		uint32_t offset[_PARALLEL_HASH];
		uint32_t init0[_PARALLEL_HASH];

		// share init among threads
		for (int p = 0; p < _PARALLEL_HASH; p++) {
			uint2 shuffle[8];
			for (int j = 0; j < 8; j++) {
				shuffle[j].x = shuffle(state[j].x, i + p);
				shuffle[j].y = shuffle(state[j].y, i + p);
			}
			switch (mix_idx) {
			case 0: mix[p] = vectorize2(shuffle[0], shuffle[1]); break;
			case 1: mix[p] = vectorize2(shuffle[2], shuffle[3]); break;
			case 2: mix[p] = vectorize2(shuffle[4], shuffle[5]); break;
			case 3: mix[p] = vectorize2(shuffle[6], shuffle[7]); break;
			}
			init0[p] = shuffle(shuffle[0].x, 0);
		}

		for (uint32_t a = 0; a < ACCESSES; a += 4) {
			int t = bfe(a, 2u, 3u);

			for (uint32_t b = 0; b < 4; b++) {
				for (int p = 0; p < _PARALLEL_HASH; p++) {
					offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % d_dag_size;
					offset[p] = shuffle(offset[p], t);
					mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
				}
			}
		}

		for (int p = 0; p < _PARALLEL_HASH; p++) {
			uint2 shuffle[4];
			uint32_t thread_mix = fnv_reduce(mix[p]);

			// update mix accross threads
			shuffle[0].x = shuffle(thread_mix, 0);
			shuffle[0].y = shuffle(thread_mix, 1);
			shuffle[1].x = shuffle(thread_mix, 2);
			shuffle[1].y = shuffle(thread_mix, 3);
			shuffle[2].x = shuffle(thread_mix, 4);
			shuffle[2].y = shuffle(thread_mix, 5);
			shuffle[3].x = shuffle(thread_mix, 6);
			shuffle[3].y = shuffle(thread_mix, 7);

			if ((i + p) == thread_id) {
				//move mix into state:
				state[8] = shuffle[0]; state[9] = shuffle[1]; state[10] = shuffle[2]; state[11] = shuffle[3];
			}
		}
	}

	// keccak_256(keccak_512(header..nonce) .. mix);
	if (cuda_swab64(keccak_f1600_final(state)) > target)
		return false;
	mix_hash[0] = state[8]; mix_hash[1] = state[9]; mix_hash[2] = state[10]; mix_hash[3] = state[11];
	return true;
}

