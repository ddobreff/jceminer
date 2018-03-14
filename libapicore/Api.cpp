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

#include "Api.h"

Api::Api(const int& port, Farm& farm): m_farm(farm)
{
	int portNumber = port;
	bool readonly = true;

	// > 0 = rw, < 0 = ro, 0 = disabled
	if (portNumber > 0)
		readonly = false;
	else if (portNumber < 0)
		portNumber = -portNumber;

	if (portNumber > 0) {
		TcpSocketServer* conn = new TcpSocketServer("0.0.0.0", portNumber);
		this->m_server = new ApiServer(conn, JSONRPC_SERVER_V2, this->m_farm, readonly);
		this->m_server->StartListening();
	}
}
