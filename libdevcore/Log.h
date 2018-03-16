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

