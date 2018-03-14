/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include <iostream>
#include <string>
#include <mutex>

extern std::string timestamp();

#define fgReset    "\x1b[0m"   // Text Reset
#define fgGray     "\x1b[37m"  // White
#define fgRed      "\x1b[91m"  // Red
#define fgLime     "\x1b[92m"  // Green
#define fgYellow   "\x1b[93m"  // Yellow
#define fgWhite    "\x1b[97m"  // White

#define loginfo(_x) \
    {std::clog << fgWhite << timestamp() << fgReset << ' ' << _x << std::endl;}
#define logwarn(_x) \
    {std::clog << fgYellow << timestamp() << fgReset << ' ' << _x << std::endl;}
#define logerror(_x) \
    {std::clog << fgRed << timestamp() << fgReset << ' ' << _x << std::endl;}

