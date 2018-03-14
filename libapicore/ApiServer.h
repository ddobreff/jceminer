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

#pragma once

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <jsonrpccpp/server.h>

using namespace jsonrpc;
using namespace dev;
using namespace dev::eth;
using namespace std::chrono;

class ApiServer : public AbstractServer<ApiServer>
{
public:
	ApiServer(AbstractServerConnector* conn, serverVersion_t type, Farm& farm, bool& readonly);
private:
	Farm& m_farm;
	void getMinerStat1(const Json::Value& request, Json::Value& response);
	void getMinerStatHR(const Json::Value& request, Json::Value& response);
	void doMinerRestart(const Json::Value& request, Json::Value& response);
	void doMinerReboot(const Json::Value& request, Json::Value& response);
};

