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

#include "Log.h"
#include <chrono>
#include <ctime>

std::mutex x_log;

std::string timestamp()
{
	time_t rawTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	char buf[24];
	if (strftime(buf, 24, "%X", localtime(&rawTime)) == 0)
		buf[0] = '\0';
	return std::string(buf);
}
