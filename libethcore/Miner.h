/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <thread>
#include <list>
#include <string>
#include <boost/timer.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/Worker.h>
#include <libdevcore/Log.h>
#include "EthashAux.h"

#define MINER_WAIT_STATE_WORK    1


#define DAG_LOAD_MODE_PARALLEL   0
#define DAG_LOAD_MODE_SEQUENTIAL 1
#define DAG_LOAD_MODE_SINGLE     2

extern bool g_logSwitchTime;
extern bool g_logJson;

using namespace std;

namespace dev
{

namespace eth
{

enum class MinerType {
	Mixed,
	CL,
	CUDA
};

enum class HwMonitorInfoType {
	UNKNOWN,
	NVIDIA,
	AMD
};

enum class HwMonitorIndexSource {
	UNKNOWN,
	OPENCL,
	CUDA
};

struct HwMonitorInfo {
	HwMonitorInfoType deviceType = HwMonitorInfoType::UNKNOWN;
	HwMonitorIndexSource indexSource = HwMonitorIndexSource::UNKNOWN;
	int deviceIndex = -1;
	string deviceName;
	string deviceId;
};

struct HwMonitor {
	int tempC = 0;
	int fanP = 0;
	double powerW = 0;
};

inline std::ostream& operator<<(std::ostream& os, HwMonitor& _hw)
{
	os << _hw.tempC << "C " << _hw.fanP << "%";
	if (_hw.powerW != 0)
		os << ' ' << fixed << setprecision(0) << _hw.powerW << "W";
	return os;
}

/// Describes the progress of a mining operation.
struct WorkingProgress {
	uint64_t hashes = 0;        ///< Total number of hashes computed.
	uint64_t ms = 0;            ///< Total number of milliseconds of mining thus far.
	uint64_t rate() const
	{
		return ms == 0 ? 0 : hashes * 1000 / ms;
	}

	std::vector<uint64_t> minersHashes;
	std::vector<HwMonitor> minerMonitors;
	uint64_t minerRate(const uint64_t hashCount) const
	{
		return ms == 0 ? 0 : hashCount * 1000 / ms;
	}
};

inline std::ostream& operator<<(std::ostream& os, WorkingProgress& _p)
{
	float mh = _p.rate() / 1000000.0f;
	os << "Speed " fgWhite << std::fixed << std::setprecision(2) << mh << fgReset " Mh/s ";
	for (size_t i = 0; i < _p.minersHashes.size(); ++i) {
		mh = _p.minerRate(_p.minersHashes[i]) / 1000000.0f;
		os << "gpu" << i << " " fgWhite << std::fixed << std::setprecision(2) << mh << fgReset;

		if (i < _p.minerMonitors.size()) {
			auto m = _p.minerMonitors[i];
			os << ' ' << m;
		}
		os << ' ';
	}

	return os;
}

class SolutionStats
{
public:
	void accepted()
	{
		accepts++;
	}
	void rejected()
	{
		rejects++;
	}
	void failed()
	{
		failures++;
	}

	void acceptedStale()
	{
		acceptedStales++;
	}

	void reset()
	{
		accepts = rejects = failures = acceptedStales = 0;
	}

	unsigned getAccepts()
	{
		return accepts;
	}
	unsigned getRejects()
	{
		return rejects;
	}
	unsigned getFailures()
	{
		return failures;
	}
	unsigned getAcceptedStales()
	{
		return acceptedStales;
	}
private:
	unsigned accepts  = 0;
	unsigned rejects  = 0;
	unsigned failures = 0;

	unsigned acceptedStales = 0;
};

inline std::ostream& operator<<(std::ostream& os, SolutionStats s)
{
	os << 'A' << s.getAccepts();
	if (s.getAcceptedStales())
		os << '+' << s.getAcceptedStales();
	if (s.getRejects())
		os << ":R" << s.getRejects();
	if (s.getFailures())
		os << ":F" << s.getFailures();
	return os;
}

class Miner;


/**
        @brief Class for hosting one or more Miners.
        @warning Must be implemented in a threadsafe manner since it will be called from multiple
        miner threads.
*/
class FarmFace
{
public:
	virtual ~FarmFace() = default;

	/**
	        @brief Called from a Miner to note a WorkPackage has a solution.
	        @param _p The solution.
	        @return true iff the solution was good (implying that mining should be .
	*/
	virtual void submitProof(Solution const& _p) = 0;
	virtual void failedSolution() = 0;
	virtual uint64_t get_nonce_scrambler() = 0;
};

/**
        @brief A miner - a member and adoptee of the Farm.
        @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
*/
#define LOG2_MAX_MINERS 5u
#define MAX_MINERS (1u << LOG2_MAX_MINERS)

class Miner: public Worker
{
public:

	Miner(std::string const& _name, FarmFace& _farm, size_t _index):
		Worker(_name + std::to_string(_index)),
		index(_index),
		farm(_farm)
	{}

	virtual ~Miner() = default;

	void setWork(WorkPackage const& _work)
	{
		{
			Guard l(x_work);
			m_work = _work;
			workSwitchStart = std::chrono::high_resolution_clock::now();
		}
		kick_miner();
	}

	uint64_t hashCount()
	{
		return m_hashCount.exchange(0, memory_order_relaxed);
	}

	unsigned Index()
	{
		return index;
	};

	HwMonitorInfo& hwmonInfo()
	{
		return m_hwmoninfo;
	}

	uint64_t get_start_nonce()
	{
		// Each GPU is given a non-overlapping 2^40 range to search
		return farm.get_nonce_scrambler() + ((uint64_t) index << 40);
	}

protected:

	virtual void kick_miner() = 0;

	WorkPackage work() const
	{
		Guard l(x_work);
		return m_work;
	}

	void addHashCount(uint32_t _n)
	{
		m_hashCount.fetch_add(_n, memory_order_relaxed);
	}

	static unsigned s_dagLoadMode;
	static unsigned s_dagLoadIndex;
	static unsigned s_dagCreateDevice;
	static uint8_t* s_dagInHostMemory;

	const size_t index = 0;
	FarmFace& farm;
	std::chrono::high_resolution_clock::time_point workSwitchStart;
	HwMonitorInfo m_hwmoninfo;
	mutable std::mutex x_work;
private:
	std::atomic<uint64_t> m_hashCount = {0};

	WorkPackage m_work;
};

}
}
