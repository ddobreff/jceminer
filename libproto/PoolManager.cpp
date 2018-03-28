/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "PoolManager.h"
#include "libdevcore/Log.h"
#include <chrono>
#include <sstream>
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using namespace dev;
using namespace eth;

static string hashToString(double diff, bool rate)
{
	static const char* k[] = {" ", " K", " M", " G", " T", " P"};
	uint32_t i = 0;
	while ((diff > 1000.0) && (i < ((sizeof(k) / sizeof(char*)) - 2))) {
		i++;
		diff = diff / 1000.0;
	}
	stringstream ss;
	ss.imbue(std::locale(""));
	ss << fixed << setprecision(1) << diff << k[i] << (rate ? "h/s" : "");
	return ss.str();
}

PoolManager::PoolManager(PoolClient& client, Farm& farm, MinerType const& minerType) :
	Worker("main"), m_client(client), m_farm(farm), m_minerType(minerType)
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
			        sport.str() << endl;
		}
		if (!m_farm.isMining()) {
			{
				Guard l(x_log);
				loginfo << "Spinning up miners..." << endl;
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
			logwarn << "Disconnected from " + m_connection.Host() << endl;
		}

		tryReconnect();
	});

	m_client.onWorkReceived([&](WorkPackage const & wp) {
		m_reconnectTry = 0;
		m_farm.setWork(wp);
		if (wp.boundary != m_lastBoundary) {
			using namespace boost::multiprecision;

			m_lastBoundary = wp.boundary;
			static const uint256_t dividend("0xffff000000000000000000000000000000000000000000000000000000000000");
			const uint256_t divisor(string("0x") + m_lastBoundary.hex());
			m_difficulty = double(dividend / divisor);
			{
				Guard l(x_log);
				loginfo << "New pool difficulty: " EthYellow << hashToString(m_difficulty / 1000000.0, false) << EthReset "\n";
			}
		}
		{
			Guard l(x_log);
			loginfo << "Received new job " << wp.header.hex().substr(0, 8) << ".." << endl;
		}
	});

	m_client.onSolutionAccepted([&](bool const & stale) {
		using namespace std::chrono;
		m_farm.acceptedSolution(stale);
		steady_clock::time_point now = steady_clock::now();
		if (!stale) {
			Guard l(x_list);
			m_accepts.push_back(now);
		}
		stringstream effRate;
		effectiveHR(effRate);
		auto ms = duration_cast<milliseconds>(now - m_submit_time);
		{
			Guard l(x_log);
			loginfo << string(stale ? EthYellow : EthLime) << "Accepted" << (stale ? " (stale)" : "") << " in " << ms.count() <<
			        " ms. " << effRate.str() << EthReset << endl;
		}
	});

	m_client.onSolutionRejected([&](bool const & stale) {
		using namespace std::chrono;
		auto ms = duration_cast<milliseconds>(steady_clock::now() - m_submit_time);
		{
			Guard l(x_log);
			loginfo << EthRed "Rejected" << (stale ? " (stale)" : "") << " in " << ms.count() << " ms." << EthReset << endl;
		}
		m_farm.rejectedSolution(stale);
	});

	m_farm.onSolutionFound([&](const std::string & from, Solution sol) {
		m_submit_time = std::chrono::steady_clock::now();
		m_client.submitSolution(sol);

		if (sol.stale) {
			Guard l(x_log);
			loginfo << EthYellow << from << " - stale nonce 0x" + toHex(sol.nonce) + " submitted to " +
			        m_connection.Host() << EthReset << endl;
		} else {
			Guard l(x_log);
			loginfo << EthWhite << from << " - nonce 0x" + toHex(sol.nonce)
			        + " submitted to " + m_connection.Host() << EthReset << endl;
		}

		return false;
	});
}

void PoolManager::effectiveHR(stringstream& ss)
{
	using namespace std::chrono;
	steady_clock::time_point now = steady_clock::now();
	auto windowStart = now - hours(6);
	{
		Guard l(x_list);
		while (m_accepts.size() && m_accepts.front() < windowStart)
			m_accepts.pop_front();
	}
	if (windowStart < m_farm.farmLaunched())
		windowStart = m_farm.farmLaunched();
	auto secs = duration_cast<seconds>(now - windowStart);
	double EHR = (m_accepts.size()  * m_difficulty) / secs.count();
	ss <<  "Effective HR (" << fixed << setprecision(2) << secs.count() / 3600.0 << "h. sma): " << hashToString(EHR, true);
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
		logerror << "Manager has no connections defined!" << endl;
	}
}

void PoolManager::tryReconnect()
{
	// No connections available, so why bother trying to reconnect
	if (m_connection.Empty()) {
		{
			Guard l(x_log);
			logerror << "Manager has no connections defined!" << endl;
		}
		return;
	}
	{
		Guard l(x_log);
		logwarn << "Retrying in 3 seconds.\n";
	}

	this_thread::sleep_for(chrono::seconds(3));

	if (m_reconnectTries >= m_reconnectTry) {
		m_reconnectTry++;
		m_client.connect();
		return;
	}

	logerror << "Could not connect.\n";
	throw runtime_error("Connect fail");
}
