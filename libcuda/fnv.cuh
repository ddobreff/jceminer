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

#define FNV_PRIME   0x01000193

__device__ __forceinline__
uint32_t fnv(uint32_t x, uint32_t y)
{
    return ((x) * FNV_PRIME ^ (y));
}

__device__ __forceinline__
uint4 fnv4(uint4 a, uint4 b)
{
    uint4 c;
    c.x = a.x * FNV_PRIME ^ b.x;
    c.y = a.y * FNV_PRIME ^ b.y;
    c.z = a.z * FNV_PRIME ^ b.z;
    c.w = a.w * FNV_PRIME ^ b.w;
    return c;
}

__device__ __forceinline__
uint32_t fnv_reduce(uint4 v)
{
    return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

