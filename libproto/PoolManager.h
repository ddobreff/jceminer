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

#include <iostream>
#include <list>
#include <libdevcore/Worker.h>
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>

#include "PoolClient.h"

using namespace std;

namespace dev
{
namespace eth
{
class PoolManager : public Worker
{
public:
	PoolManager(PoolClient& client, Farm& farm, MinerType const& minerType);
	void addConnection(PoolConnection& conn);
	void start();
	void setReconnectTries(unsigned const& reconnectTries)
	{
		m_reconnectTries = reconnectTries;
	};
	bool isConnected()
	{
		return m_client.isConnected();
	};
	bool difficulty()
	{
		return m_difficulty;
	};

private:
	void tryReconnect();
	void workLoop() override;

	PoolClient& m_client;
	unsigned m_reconnectTries = 3;
	unsigned m_reconnectTry = 0;
	unsigned m_activeConnectionIdx = 0;
	PoolConnection m_connection;
	h256 m_lastBoundary = h256();
	Farm& m_farm;
	MinerType m_minerType;
	std::chrono::steady_clock::time_point m_submit_time;
	list<std::chrono::steady_clock::time_point> m_accepts;
	double m_difficulty;
	bool m_farmStarted = false;
};
}
}

