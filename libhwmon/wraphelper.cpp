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

#include "wraphelper.h"

/* assume we can use dlopen itself... */
void* wrap_dlopen(const char* filename)
{
	return dlopen(filename, RTLD_NOW);
}
void* wrap_dlsym(void* h, const char* sym)
{
	return dlsym(h, sym);
}
int wrap_dlclose(void* h)
{
	return dlclose(h);
}
