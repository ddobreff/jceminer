/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "Log.h"
#include <chrono>
#include <ctime>

std::mutex xLogMtx;
std::locale logLocale = std::locale("");

std::string timestamp()
{
	using std::chrono::system_clock;
	time_t rawTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	char buf[24];
	strftime(buf, 24, "%X", localtime(&rawTime));
	return std::string(buf);
}
