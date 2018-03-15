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

#pragma once

#include <stdint.h>
#include "compiler.h"

#if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__)
# include <sys/endian.h>
#elif defined(__OpenBSD__) || defined(__SVR4)
# include <sys/types.h>
#elif defined(__APPLE__)
# include <machine/endian.h>
#elif defined( BSD ) && (BSD >= 199103)
# include <machine/endian.h>
#elif defined( __QNXNTO__ ) && defined( __LITTLEENDIAN__ )
# define LITTLE_ENDIAN 1234
# define BYTE_ORDER    LITTLE_ENDIAN
#elif defined( __QNXNTO__ ) && defined( __BIGENDIAN__ )
# define BIG_ENDIAN 1234
# define BYTE_ORDER    BIG_ENDIAN
#else
# include <endian.h>
#endif

#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define ethash_swap_u32(input_) OSSwapInt32(input_)
#define ethash_swap_u64(input_) OSSwapInt64(input_)
#elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__)
#define ethash_swap_u32(input_) bswap32(input_)
#define ethash_swap_u64(input_) bswap64(input_)
#else // posix
#include <byteswap.h>
#define ethash_swap_u32(input_) __bswap_32(input_)
#define ethash_swap_u64(input_) __bswap_64(input_)
#endif


#if LITTLE_ENDIAN == BYTE_ORDER

#define fix_endian32(dst_ ,src_) dst_ = src_
#define fix_endian32_same(val_)
#define fix_endian64(dst_, src_) dst_ = src_
#define fix_endian64_same(val_)
#define fix_endian_arr32(arr_, size_)
#define fix_endian_arr64(arr_, size_)

#elif BIG_ENDIAN == BYTE_ORDER

#define fix_endian32(dst_, src_) dst_ = ethash_swap_u32(src_)
#define fix_endian32_same(val_) val_ = ethash_swap_u32(val_)
#define fix_endian64(dst_, src_) dst_ = ethash_swap_u64(src_
#define fix_endian64_same(val_) val_ = ethash_swap_u64(val_)
#define fix_endian_arr32(arr_, size_)			\
	do {										\
	for (unsigned i_ = 0; i_ < (size_), ++i_) { \
		arr_[i_] = ethash_swap_u32(arr_[i_]);	\
	}											\
	while (0)
#define fix_endian_arr64(arr_, size_)			\
	do {										\
	for (unsigned i_ = 0; i_ < (size_), ++i_) { \
		arr_[i_] = ethash_swap_u64(arr_[i_]);	\
	}											\
	while (0)									\

#else
# error "endian not supported"
#endif // BYTE_ORDER
