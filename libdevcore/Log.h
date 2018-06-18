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

extern std::mutex xLogMtx;
extern std::locale logLocale;

#define logLevel(_x, _l) \
{ \
    std::stringstream ss; \
    ss.imbue(logLocale); \
    ss << _l << timestamp() << fgReset << ' ' << _x << std::endl; \
    std::clog << ss.str(); \
}

#define loginfo(_x) logLevel(_x, fgWhite)
#define logwarn(_x) logLevel(_x, fgYellow)
#define logerror(_x) logLevel(_x, fgRed)

