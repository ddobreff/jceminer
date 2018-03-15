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

#include "PoolManager.h"
#include "libdevcore/Log.h"
#include <chrono>
#include <sstream>

using namespace std;
using namespace dev;
using namespace eth;

static string hashToString(double diff, bool rate)
{
	static const char* k[] = {" h", " Kh", " Mh", " Gh", " Th", " Ph"};
	uint32_t i = 0;
	while ((diff > 1000.0) && (i < ((sizeof(k) / sizeof(char*)) - 2))) {
		i++;
		diff = diff / 1000.0;
	}
	stringstream ss;
	ss.imbue(std::locale(""));
	ss << fixed << setprecision(1) << diff << k[i] << (rate ? "/s" : "");
	return ss.str();
}

PoolManager::PoolManager(PoolClient& client, Farm& farm, MinerType const& minerType) : Worker("main"), m_client(client),
	m_farm(farm), m_minerType(minerType)
{

	m_client.onConnected([&](boost::asio::ip::address address) {
		m_connection.Address(address);
		stringstream saddr;
		const boost::asio::ip::address nulladdr;
		if (address != nulladdr)
			saddr << '(' << address << ')';
		stringstream sport;
		sport << m_connection.Port();
		{
			Guard l(x_log);
			loginfo << "Connected to " << m_connection.Host() << saddr.str() << ':' <<
			        sport.str() << endl << flush;
		}
		if (!m_farm.isMining()) {
			{
				Guard l(x_log);
				loginfo << "Spinning up miners..." << endl << flush;
			}
			if (m_minerType == MinerType::CL)
				m_farm.start("opencl", false);
			else if (m_minerType == MinerType::CUDA)
				m_farm.start("cuda", false);
			else if (m_minerType == MinerType::Mixed) {
				m_farm.start("cuda", false);
				m_farm.start("opencl", true);
			}
			m_farmStarted = true;
		}
	});

	m_client.onDisconnected([&]() {
		{
			Guard l(x_log);
			logwarn << "Disconnected from " + m_connection.Host() << endl << flush;
		}

		tryReconnect();
	});

	m_client.onWorkReceived([&](WorkPackage const & wp) {
		m_reconnectTry = 0;
		m_farm.setWork(wp);
		if (wp.boundary != m_lastBoundary) {
			using namespace boost::multiprecision;

			m_lastBoundary = wp.boundary;
			static const uint512_t dividend("0x10000000000000000000000000000000000000000000000000000000000000000");
			const uint256_t divisor(string("0x") + m_lastBoundary.hex());
			m_difficulty = double(dividend / divisor);
			{
				Guard l(x_log);
				loginfo << "New pool difficulty: " << hashToString(m_difficulty, false) << endl << flush;
			}
		}
		{
			Guard l(x_log);
			loginfo << "Received new job " << wp.header.hex().substr(0, 8) << ".." << endl << flush;
		}
	});

	m_client.onSolutionAccepted([&](bool const & stale) {
		using namespace std::chrono;

		m_farm.acceptedSolution(stale);
		std::chrono::steady_clock::time_point now = steady_clock::now();
		if (!stale)
			m_accepts.push_back(now);
		auto windowStart = now - hours(6);
		while (m_accepts.size() && m_accepts.front() < windowStart)
			m_accepts.pop_front();
		if (windowStart < m_farm.farmLaunched())
			windowStart = m_farm.farmLaunched();
		auto ms = duration_cast<milliseconds>(now - m_submit_time);
		auto secs = duration_cast<seconds>(now - windowStart);
		double EHR = (m_accepts.size()  * m_difficulty) / secs.count();
		{
			Guard l(x_log);
			loginfo << string(stale ? EthYellow : EthLime) << "Accepted" << (stale ? " (stale)" : "") << " in " << ms.count() <<
			        " ms." << " Effective HR (" << fixed << setprecision(2) << secs.count() / 3600.0 << "h. sma): " << hashToString(EHR,
			                true) << EthReset << endl << flush;
		}
	});

	m_client.onSolutionRejected([&](bool const & stale) {
		using namespace std::chrono;
		auto ms = duration_cast<milliseconds>(steady_clock::now() - m_submit_time);
		{
			Guard l(x_log);
			loginfo << EthRed "Rejected" << (stale ? " (stale)" : "") << " in " << ms.count() << " ms." << EthReset << endl <<
			        flush;
		}
		m_farm.rejectedSolution(stale);
	});

	m_farm.onSolutionFound([&](const std::string & from, Solution sol) {
		m_submit_time = std::chrono::steady_clock::now();
		m_client.submitSolution(sol);

		if (sol.stale) {
			Guard l(x_log);
			loginfo << EthYellow << from << " - stale nonce 0x" + toHex(sol.nonce) + " submitted to " +
			        m_connection.Host() << EthReset << endl << flush;
		} else {
			Guard l(x_log);
			loginfo << EthWhite << from << " - nonce 0x" + toHex(sol.nonce)
			        + " submitted to " + m_connection.Host() << EthReset << endl << flush;
		}

		return false;
	});
}

void PoolManager::workLoop()
{
	while (true) {
		this_thread::sleep_for(chrono::minutes(2));
		// Hashrate reporting
		if (m_farmStarted) {
			auto mp = m_farm.miningProgress();
			m_client.submitHashrate(mp.rate());
		}
	}
}

void PoolManager::addConnection(PoolConnection& conn)
{
	if (conn.Empty())
		return;

	m_connection = conn;

	m_client.setConnection(conn);
	m_farm.set_pool_addresses(conn.Host(), conn.Port());
}

void PoolManager::start()
{
	if (!m_connection.Empty()) {
		startWorking();
		// Try to connect to pool
		m_client.connect();
	} else {
		Guard l(x_log);
		logerror << "Manager has no connections defined!" << endl << flush;
	}
}

void PoolManager::tryReconnect()
{
	// No connections available, so why bother trying to reconnect
	if (m_connection.Empty()) {
		{
			Guard l(x_log);
			logerror << "Manager has no connections defined!" << endl << flush;
		}
		return;
	}

	{
		Guard l(x_log);
		logwarn << "Retrying in 3 seconds.\n" << flush;
	}
	this_thread::sleep_for(chrono::seconds(3));

	// Fallback logic, tries current connection multiple times and then switches to
	// one of the other connections.
	if (m_reconnectTries >= m_reconnectTry) {
		m_reconnectTry++;
		m_client.connect();
	} else
		BOOST_THROW_EXCEPTION(StratumTimeout());
}
