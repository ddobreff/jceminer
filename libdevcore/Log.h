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

#define loginfo(_x) \
	{x_log.lock(); std::clog << EthWhite << timestamp() << EthReset << ' ' << _x << std::endl; x_log.unlock();}
#define logwarn(_x) \
	{x_log.lock(); std::clog << EthYellow << timestamp() << EthReset << ' ' << _x << std::endl; x_log.unlock();}
#define logerror(_x) \
	{x_log.lock(); std::clog << EthRed << timestamp() << EthReset << ' ' << _x << std::endl; x_log.unlock();}

