/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

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
