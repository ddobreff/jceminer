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
	static const char* k[] = {"", "K", "M", "G", "T", "P"};
	uint32_t i = 0;
	while ((diff > 1000.0) && (i < ((sizeof(k) / sizeof(char*)) - 2))) {
		i++;
		diff = diff / 1000.0;
	}
	stringstream ss;
	ss.imbue(std::locale(""));
	ss << fixed << setprecision(2) << diff << ' ' << k[i] << (rate ? "h/s" : "");
	return ss.str();
}

extern bool g_display_effective;

PoolManager::PoolManager(PoolClient& client, Farm& farm, MinerType const& minerType) :
	Worker("main"), m_client(client), m_farm(farm), m_minerType(minerType), m_effStartTime(farm.farmLaunched())
{

	m_client.onConnected([&](boost::asio::ip::address address) {
		m_connection.Address(address);
		stringstream saddr;
		const boost::asio::ip::address nulladdr;
		if (address != nulladdr)
			saddr << '(' << address << ')';
		stringstream sport;
		sport << m_connection.Port();
		loginfo("Connected to " << m_connection.Host() << saddr.str() << ':' << sport.str());
		if (!m_farm.isMining()) {
			loginfo("Spinning up miners...");
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
		logwarn("Disconnected from " + m_connection.Host());

		tryReconnect();
	});

	m_client.onWorkReceived([&](WorkPackage const & wp) {
		m_reconnectTry = 0;
		m_farm.setWork(wp);
		if (wp.boundary != m_lastBoundary) {
			using namespace boost::multiprecision;

			if (m_lastBoundary != wp.boundary) {
				if (g_display_effective) {
					Guard l(x_list);
					m_effStartTime = std::chrono::steady_clock::now();
					m_10_accepts.clear();
					m_60_accepts.clear();
					m_360_accepts.clear();
				}
				m_lastBoundary = wp.boundary;
				static const uint256_t dividend("0xffff000000000000000000000000000000000000000000000000000000000000");
				const uint256_t divisor(string("0x") + m_lastBoundary.hex());
				m_difficulty = double(dividend / divisor);
				loginfo("Difficulty: " fgYellow << hashToString(m_difficulty / 1000000.0, false) << fgReset);
			}
		}
		loginfo("Header: " fgWhite "0x" << wp.header.hex().substr(0, 15) << ".." fgReset);
	});

	m_client.onSolutionAccepted([&](bool stale) {
		using namespace std::chrono;
		m_farm.acceptedSolution(stale);
		steady_clock::time_point now = steady_clock::now();
		auto ms = duration_cast<milliseconds>(now - m_submit_time);
		if (!stale && g_display_effective) {
			stringstream effRate;
			{
				Guard l(x_list);
				m_10_accepts.push_back(now);
				effectiveHR(effRate);
			}
			loginfo(effRate.str());
		}
		loginfo(string(stale ? fgYellow : fgLime) << "Accepted" << (stale ? " (stale)" : "") << " in " << ms.count() <<
		        " ms. " << fgReset);
	});

	m_client.onSolutionRejected([&](bool stale, string const & msg) {
		using namespace std::chrono;
		auto ms = duration_cast<milliseconds>(steady_clock::now() - m_submit_time);
		loginfo(fgRed "Rejected" << (stale ? " (stale)" : "") << " in " << ms.count() << " ms." << fgReset << " " << msg);
		m_farm.rejectedSolution();
	});

	m_farm.onSolutionFound([&](Solution sol) {
		m_submit_time = std::chrono::steady_clock::now();
		m_client.submitSolution(sol);
		loginfo(string(sol.stale ? fgYellow : fgWhite) << sol.gpu << (sol.stale ? " (stale)" : "") << " 0x" + toHex(
		            sol.nonce) + " submitted" << fgReset);
		return false;
	});
}

void PoolManager::effectiveHR(stringstream& ss)
{
	using namespace std::chrono;
	steady_clock::time_point now = steady_clock::now();
	auto windowStart10 = now - minutes(10);
	auto windowStart60 = now - hours(1);
	auto windowStart360 = now - hours(6);
	size_t s10, s60, s360;
	{
		while (m_10_accepts.size() && (m_10_accepts.front() < windowStart10)) {
			m_60_accepts.push_back(m_10_accepts.front());
			m_10_accepts.pop_front();
		}
		while (m_60_accepts.size() && (m_60_accepts.front() < windowStart60)) {
			m_360_accepts.push_back(m_60_accepts.front());
			m_60_accepts.pop_front();
		}
		while (m_360_accepts.size() && (m_360_accepts.front() < windowStart360))
			m_360_accepts.pop_front();
		s10 = m_10_accepts.size();
		s60 = m_60_accepts.size();
		s360 = m_360_accepts.size();
	}

	if (windowStart10 < m_effStartTime)
		windowStart10 = m_effStartTime;
	auto secs = duration_cast<seconds>(now - windowStart10);
	double EHR = (s10 * m_difficulty) / secs.count();
	ss <<  "Eff.HR " << fixed << setprecision(2) << secs.count() / 60.0 << " min. sma @ " << hashToString(EHR, true);
	if (s60) {
		if (windowStart60 < m_effStartTime)
			windowStart60 = m_effStartTime;
		secs = duration_cast<seconds>(now - windowStart60);
		EHR = ((s60 + s10)  * m_difficulty) / secs.count();
		ss <<  ", " << fixed << setprecision(2) << secs.count() / 3600.0 << " hour sma @ " << hashToString(EHR, true);
		if (s360) {
			if (windowStart360 < m_effStartTime)
				windowStart360 = m_effStartTime;
			secs = duration_cast<seconds>(now - windowStart360);
			EHR = ((s360 + s60 + s10) * m_difficulty) / secs.count();
			ss <<  ", " << fixed << setprecision(2) << secs.count() / 3600.0 << " hour sma @ " << hashToString(EHR, true);
		}
	}
}

extern bool g_report_stratum_hashrate;

void PoolManager::workLoop()
{
	while (true) {
		this_thread::sleep_for(chrono::minutes(2));
		// Hashrate reporting
		if (m_farmStarted) {
			auto mp = m_farm.miningProgress();
			if (g_report_stratum_hashrate)
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
	}
	else
		logerror("Manager has no connections defined!");
}

void PoolManager::tryReconnect()
{
	// No connections available, so why bother trying to reconnect
	if (m_connection.Empty()) {
		logerror("Manager has no connections defined!");
		return;
	}
	logwarn("Retrying in 3 seconds.");

	this_thread::sleep_for(chrono::seconds(3));

	if (m_reconnectTries >= m_reconnectTry) {
		m_reconnectTry++;
		m_client.connect();
		return;
	}

	logerror("Could not connect.");
	throw runtime_error("Connect fail");
}
