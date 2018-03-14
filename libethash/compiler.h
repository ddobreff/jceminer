/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

// Visual Studio doesn't support the inline keyword in C mode
#if defined(_MSC_VER) && !defined(__cplusplus)
#define inline __inline
#endif

// pretend restrict is a standard keyword
#if defined(_MSC_VER)
#define restrict __restrict
#else
#define restrict __restrict__
#endif

