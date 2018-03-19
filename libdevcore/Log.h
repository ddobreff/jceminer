/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include <string>
#include <mutex>

extern std::string timestamp();

#define EthReset	"\x1b[0m"	// Text Reset
#define EthGray		"\x1b[37m"	// White
#define EthRed		"\x1b[91m"	// Red
#define EthLime		"\x1b[92m"	// Green
#define EthYellow	"\x1b[93m"	// Yellow
#define EthWhite	"\x1b[97m"	// White

extern std::mutex x_log;

#define loginfo std::clog << EthWhite << timestamp() << EthReset << ' '
#define logwarn std::clog << EthYellow << timestamp() << EthReset << ' '
#define logerror std::clog << EthRed << timestamp() << EthReset << ' '

