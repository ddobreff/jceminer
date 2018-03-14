// This source code is licenced under GNU General Public License, Version 3.

#pragma once

#include <stdint.h>
#include "compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FNV_PRIME 0x01000193

static inline uint32_t fnv_hash(uint32_t const x, uint32_t const y)
{
	return x * FNV_PRIME ^ y;
}

#ifdef __cplusplus
}
#endif
