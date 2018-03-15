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

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <thread>
#include <list>
#include <libdevcore/Common.h>
#include <libdevcore/Worker.h>
#include <libethcore/Miner.h>
#include <libhwmon/wrapnvml.h>
#include <libhwmon/wrapadl.h>
#include <libhwmon/wrapamdsysfs.h>

namespace dev
{

namespace eth
{

class Farm: public FarmFace
{
public:
	struct SealerDescriptor {
		std::function<unsigned()> instances;
		std::function<Miner*(FarmFace&, unsigned)> create;
	};

	Farm(): m_hashrateTimer(m_io_service)
	{
		// Given that all nonces are equally likely to solve the problem
		// we could reasonably always start the nonce search ranges
		// at a fixed place, but that would be boring. Provide a once
		// per run randomized start place, without creating much overhead.
		random_device engine;
		m_nonce_scrambler = uniform_int_distribution<uint64_t>()(engine);

		// Init HWMON
		adlh = wrap_adl_create();
		sysfsh = wrap_amdsysfs_create();
		nvmlh = wrap_nvml_create();
	}

	~Farm()
	{
		// Deinit HWMON
		if (adlh)
			wrap_adl_destroy(adlh);
		if (sysfsh)
			wrap_amdsysfs_destroy(sysfsh);
		if (nvmlh)
			wrap_nvml_destroy(nvmlh);
	}

	void setWork(WorkPackage const& _wp)
	{
		// Set work to each miner
		Guard l(x_minerWork);
		if (_wp.header == m_work.header && _wp.startNonce == m_work.startNonce)
			return;
		m_work = _wp;
		for (auto const& m : m_miners)
			m->setWork(m_work);
	}

	void setSealers(std::map<std::string, SealerDescriptor> const& _sealers)
	{
		m_sealers = _sealers;
	}

	bool start(std::string const& _sealer, bool mixed)
	{
		Guard l(x_minerWork);
		if (!m_miners.empty() && m_lastSealer == _sealer)
			return true;
		if (!m_sealers.count(_sealer))
			return false;

		if (!mixed)
			m_miners.clear();
		auto ins = m_sealers[_sealer].instances();
		unsigned start = 0;
		if (!mixed)
			m_miners.reserve(ins);
		else {

			start = m_miners.size();
			ins += start;
			m_miners.reserve(ins);
		}
		for (unsigned i = start; i < ins; ++i) {
			// TODO: Improve miners creation, use unique_ptr.
			m_miners.push_back(std::shared_ptr<Miner>(m_sealers[_sealer].create(*this, i)));

			// Start miners' threads. They should pause waiting for new work
			// package.
			m_miners.back()->startWorking();
		}
		m_isMining = true;
		m_lastSealer = _sealer;
		b_lastMixed = mixed;

		// Start hashrate collector
		m_hashrateTimer.cancel();
		m_hashrateTimer.expires_from_now(boost::posix_time::milliseconds(1000));
		m_hashrateTimer.async_wait(boost::bind(&Farm::processHashRate, this, boost::asio::placeholders::error));

		if (m_serviceThread.joinable()) {
			m_io_service.reset();
			m_serviceThread.join();
		}

		m_serviceThread = std::thread{ boost::bind(&boost::asio::io_service::run, &m_io_service) };

		return true;
	}

	void processHashRate(const boost::system::error_code& ec)
	{

		if (!ec) {
			auto now = std::chrono::steady_clock::now();

			Guard l(x_minerWork);

			m_progress.ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastStart).count();
			m_lastStart = now;

			m_progress.minersHashes.clear();
			m_progress.hashes = 0;
			for (auto const& i : m_miners) {
				uint64_t minerHashCount = i->hashCount();
				m_progress.hashes += minerHashCount;
				m_progress.minersHashes.push_back(minerHashCount);
			}

			// Restart timer
			m_hashrateTimer.expires_from_now(boost::posix_time::milliseconds(1000));
			m_hashrateTimer.async_wait(boost::bind(&Farm::processHashRate, this, boost::asio::placeholders::error));
		}
	}

	bool isMining() const
	{
		return m_isMining;
	}

	WorkingProgress const& miningProgress(bool hwmon = false, bool power = false) const
	{
		Guard l(x_minerWork);
		m_progress.minerMonitors.clear();
		for (auto const& i : m_miners) {
			if (hwmon) {
				HwMonitorInfo hwInfo = i->hwmonInfo();
				HwMonitor hw;
				unsigned int tempC = 0, fanpcnt = 0, powerW = 0;
				if (hwInfo.deviceIndex >= 0) {
					if (hwInfo.deviceType == HwMonitorInfoType::NVIDIA && nvmlh) {
						int typeidx = 0;
						if (hwInfo.indexSource == HwMonitorIndexSource::CUDA)
							typeidx = nvmlh->cuda_nvml_device_id[hwInfo.deviceIndex];
						else if (hwInfo.indexSource == HwMonitorIndexSource::OPENCL)
							typeidx = nvmlh->opencl_nvml_device_id[hwInfo.deviceIndex];
						else {
							//Unknown, don't map
							typeidx = hwInfo.deviceIndex;
						}
						wrap_nvml_get_tempC(nvmlh, typeidx, &tempC);
						wrap_nvml_get_fanpcnt(nvmlh, typeidx, &fanpcnt);
						if (power)
							wrap_nvml_get_power_usage(nvmlh, typeidx, &powerW);
					} else if (hwInfo.deviceType == HwMonitorInfoType::AMD && adlh) {
						int typeidx = 0;
						if (hwInfo.indexSource == HwMonitorIndexSource::OPENCL)
							typeidx = adlh->opencl_adl_device_id[hwInfo.deviceIndex];
						else {
							//Unknown, don't map
							typeidx = hwInfo.deviceIndex;
						}
						wrap_adl_get_tempC(adlh, typeidx, &tempC);
						wrap_adl_get_fanpcnt(adlh, typeidx, &fanpcnt);
						if (power)
							wrap_adl_get_power_usage(adlh, typeidx, &powerW);
					}
					// Overwrite with sysfs data if present
					if (hwInfo.deviceType == HwMonitorInfoType::AMD && sysfsh) {
						int typeidx = 0;
						if (hwInfo.indexSource == HwMonitorIndexSource::OPENCL)
							typeidx = sysfsh->opencl_sysfs_device_id[hwInfo.deviceIndex];
						else {
							//Unknown, don't map
							typeidx = hwInfo.deviceIndex;
						}
						wrap_amdsysfs_get_tempC(sysfsh, typeidx, &tempC);
						wrap_amdsysfs_get_fanpcnt(sysfsh, typeidx, &fanpcnt);
						if (power)
							wrap_amdsysfs_get_power_usage(sysfsh, typeidx, &powerW);
					}
				}
				hw.tempC = tempC;
				hw.fanP = fanpcnt;
				hw.powerW = powerW / ((double)1000.0);
				m_progress.minerMonitors.push_back(hw);
			}
		}

		return m_progress;
	}

	SolutionStats getSolutionStats()
	{
		return m_solutionStats;
	}

	void failedSolution() override
	{
		m_solutionStats.failed();
	}

	void acceptedSolution(bool _stale)
	{
		if (!_stale)
			m_solutionStats.accepted();
		else
			m_solutionStats.acceptedStale();
	}

	void rejectedSolution(bool _stale)
	{
		if (!_stale)
			m_solutionStats.rejected();
		else
			m_solutionStats.rejectedStale();
	}

	using SolutionFound = std::function<void(const std::string&, Solution const&)>;

	void onSolutionFound(SolutionFound const& _handler)
	{
		m_onSolutionFound = _handler;
	}

	WorkPackage work() const
	{
		Guard l(x_minerWork);
		return m_work;
	}

	std::chrono::steady_clock::time_point farmLaunched()
	{
		return m_farm_launched;
	}

	string farmLaunchedFormatted()
	{
		auto d = std::chrono::steady_clock::now() - m_farm_launched;
		int hsize = 3;
		auto hhh = std::chrono::duration_cast<std::chrono::hours>(d);
		if (hhh.count() < 100)
			hsize = 2;
		d -= hhh;
		auto mm = std::chrono::duration_cast<std::chrono::minutes>(d);
		std::ostringstream stream;
		stream << "T:" << std::setfill('0') << std::setw(hsize) << hhh.count() << ':' << std::setfill('0') << std::setw(
		           2) << mm.count();
		return stream.str();
	}

	void set_pool_addresses(string host, unsigned port)
	{
		stringstream ssPoolAddresses;
		ssPoolAddresses << host << ':' << port;
		m_pool_addresses = ssPoolAddresses.str();
	}

	string get_pool_addresses()
	{
		return m_pool_addresses;
	}

	uint64_t get_nonce_scrambler() override
	{
		return m_nonce_scrambler;
	}

private:
	void submitProof(const std::string& _from, Solution const& _s) override
	{
		assert(m_onSolutionFound);
		m_onSolutionFound(_from, _s);
	}

	mutable std::mutex x_minerWork;
	std::vector<std::shared_ptr<Miner>> m_miners;
	WorkPackage m_work;
	bool m_isMining = false;
	mutable WorkingProgress m_progress;
	SolutionFound m_onSolutionFound;
	std::map<std::string, SealerDescriptor> m_sealers;
	std::string m_lastSealer;
	bool b_lastMixed = false;
	std::chrono::steady_clock::time_point m_lastStart;
	int m_hashrateSmoothInterval = 10000;
	std::thread m_serviceThread;  ///< The IO service thread.
	boost::asio::io_service m_io_service;
	boost::asio::deadline_timer m_hashrateTimer;
	mutable SolutionStats m_solutionStats;
	std::chrono::steady_clock::time_point m_farm_launched = std::chrono::steady_clock::now();
	string m_pool_addresses;
	uint64_t m_nonce_scrambler;
	wrap_nvml_handle* nvmlh = NULL;
	wrap_adl_handle* adlh = NULL;
	wrap_amdsysfs_handle* sysfsh = NULL;
};

}
}
