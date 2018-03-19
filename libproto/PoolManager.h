/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

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

