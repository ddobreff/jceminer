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

__constant__ uint32_t d_dag_size;
__constant__ hash128_t* d_dag;
__constant__ uint32_t d_light_size;
__constant__ hash64_t* d_light;
__constant__ hash32_t d_header;
__constant__ uint64_t d_target;

