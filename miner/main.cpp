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

#include <thread>
#include <fstream>
#include <iostream>
#include <miner-buildinfo.h>
#include <chrono>
#include <fstream>
#include <random>
#include <list>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>

#include <libethcore/Exceptions.h>
#include <libdevcore/SHA3.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Farm.h>
#if ETH_ETHASHCL
#include <libcl/CLMiner.h>
#endif
#if ETH_ETHASHCUDA
#include <libcuda/CUDAMiner.h>
#endif
#include <libproto/PoolManager.h>
#include <libproto/stratum/EthStratumClient.h>
#if API_CORE
#include <libapicore/Api.h>
#endif
#include <libdevcore/Log.h>

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace boost::algorithm;

extern bool g_logSwitchTime;
extern bool g_logJson;

bool g_report_stratum_hashrate = false;
string g_email;
unsigned g_worktimeout = 180;

class BadArgument: public Exception {};

string version()
{
	auto* bi = miner_get_buildinfo();
	stringstream ss;
	ss << string("miner version ") + bi->project_version + "+git." + string(bi->git_commit_hash).substr(0, 7);
	return ss.str();
}

class MinerCLI
{
public:
	enum class OperationMode {
		None,
		Benchmark,
		Simulation,
		Farm,
		Stratum
	};

	MinerCLI() {}

	void interpretOption(int argc, char** argv)
	{
		namespace po = boost::program_options;

		po::options_description desc("Options");
		desc.add_options()
		("help,h", po::bool_switch()->default_value(false), "produce help message.")
		("devices,d", po::bool_switch()->default_value(false), "List devices.")
		("version,v", po::bool_switch()->default_value(false), "list version.")
		("retries,r", po::value<unsigned>(&m_maxFarmRetries)->default_value(3), "Connection retries.")
		("email,e", po::value<string>(&g_email), "Stratum email.")
		("timeout,w", po::value<unsigned>(&g_worktimeout)->default_value(180), "Work timeout.")
		("hash", po::bool_switch()->default_value(false), "Report hashrate to pool.")
		("stats-intvl,s", po::value<unsigned>(&m_displayInterval)->default_value(15), "statistics display interval.")
		("stats-level,l", po::value<unsigned>(&m_show_level)->default_value(0),
		 "statistics display interval. 0 - HR only, 1 - + fan & temp, 2 - + power.")
		("pool,p", po::value<string>(&m_endpoint_url), "Pool URL.\n"
		 "URL takes the form: scheme://[user[:password]@]hostname:port.\n"
		 "unsecured schemes:    stratum+tcp stratum1+tcp stratum2+tcp\n"
		 "secured with any TLS: stratum+tls stratum1+tls stratum2+tls stratum+ssl stratum1+ssl stratum2+ssl\n"
		 "secured with TLS 1.2: stratum+tls12 stratum1+tls12 stratum2+tls12\n"
		 "Example: stratum+ssl://0x012345678901234567890234567890123.miner1@ethermine.org:5555\n"
		 "Stratum versions:\n"
		 "stratum  - official stratum spec: ethpool, ethermine, coinotron, mph, nanopool\n"
		 "stratum1 - eth-proxy compatible: dwarfpool, f2pool, nanopool (required for hashrate reporting to work with nanopool)\n"
		 "stratum2 - EthereumStratum/1.0.0: nicehash\n\n")
#if API_CORE
		("api-port,a", po::value<unsigned>(&m_api_port)->default_value(80), "API port number.")
#endif
#if ETH_ETHASHCL
		("cl-plat", po::value<unsigned>(&m_openclPlatform), "Opencl platform.")
		("cl-devs", po::value<std::vector<unsigned>>()->multitoken(), "Opencl device list.")
		("cl-parallel", po::value<unsigned>(&m_openclThreadsPerHash), "Opencl parallel hashes.")
		("cl-kernel", po::value<unsigned>(&m_openclSelectedKernel), "Opencl kernel. 0 - Stable, 1 - Experimental, 2 - binary.")
		("cl-tweak", po::value<unsigned>(&m_openclWavetweak), "Opencl wave tweak.")
		("cl-global", po::value<unsigned>(&m_globalWorkSizeMultiplier), "Opencl global work size. 0 - Auto.")
		("cl-local", po::value<unsigned>(&m_localWorkSize), "Opencl local work size.")
#endif
#if ETH_ETHASHCUDA
		("cu-grid", po::value<unsigned>(&m_cudaGridSize), "Cuda grid size.")
		("cu-block", po::value<unsigned>(&m_cudaBlockSize), "Cuda block size.")
		("cu-devs", po::value<std::vector<unsigned>>()->multitoken(), "Cuda device list.")
		("cu-parallel", po::value<unsigned>(&m_parallelHash), "Cuda parallel hashes.")
		("cu-sched", po::value<unsigned>(&m_cudaSchedule), "Cuda schedule mode. 0 - auto, 1 - spin, 2 - yield, 4 - sync")
		("cu-stream", po::value<unsigned>(&m_numStreams), "Cuda streams")
		("cu-noeval", po::bool_switch()->default_value(false), "Cuda bypass software result evaluation.")
#endif
		("dag-mode", po::value<unsigned>(&m_dagLoadMode), "DAG load mode. 0 - parallel, 1 - sequential, 2 - single.")
		("log-switch", po::bool_switch()->default_value(false), "Log job switch time.")
		("log-json", po::bool_switch()->default_value(false), "Log formatted json messaging.")
		("cl,G", po::bool_switch()->default_value(false), "Opencl mode.") // set m_minerType = MinerType::CL;
		("cu,U", po::bool_switch()->default_value(false), "Cuda mode.") // set m_minerType = MinerType::CUDA;
		("mixed,X", po::bool_switch()->default_value(false),
		 "Mixed opencl and cuda mode. Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --cl-platform 1 or 2.")
		;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm["help"].as<bool>()) {
			cout << desc << "\n";
			exit(0);
		}

		if (vm["version"].as<bool>()) {
			cout << version() << "\n";
			exit(0);
		}

		if (vm["devices"].as<bool>()) {
			m_shouldListDevices = true;
			return;
		}

		if (vm.count("pool") != 1) {
			cerr << "Specify a single pool URL\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}

		string url = vm["pool"].as<string>();
		URI uri;
		try {
			uri = url;
		} catch (...) {
			cerr << "Bad endpoint address: " << url << endl;
			BOOST_THROW_EXCEPTION(BadArgument());
		}
		if (!uri.KnownScheme()) {
			cerr << "Unknown URI scheme " << uri.Scheme() << endl;
			BOOST_THROW_EXCEPTION(BadArgument());
		}
		m_endpoint = PoolConnection(uri);

		if (vm.find("cl-devs") != vm.end()) {
			m_openclDeviceCount = vm["cl-devices"].as<vector<unsigned>>().size();
			m_openclDevices = vm["cl-devices"].as<vector<unsigned>>();
		}

		if (vm.find("cu-devs") != vm.end()) {
			m_cudaDeviceCount = vm["cu-devices"].as<vector<unsigned>>().size();
			m_cudaDevices = vm["cu-devices"].as<vector<unsigned>>();
		}
		if ((m_openclDeviceCount + m_cudaDeviceCount) > MAX_GPUS) {
			cerr << "Can only support up to " << MAX_GPUS << ".\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}

		m_cudaNoEval = vm["cu-noeval"].as<bool>();

		g_logSwitchTime = vm["log-switch"].as<bool>();

		g_logJson = vm["log-json"].as<bool>();

		g_report_stratum_hashrate = vm["hash"].as<bool>();

		if (vm["cl"].as<bool>())
			m_minerType = MinerType::CL;
		else if (vm["cu"].as<bool>())
			m_minerType = MinerType::CUDA;
		else if (vm["mixed"].as<bool>())
			m_minerType = MinerType::Mixed;
		else {
			cerr << "Specify a miner type\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}

		if (m_openclThreadsPerHash != 1 && m_openclThreadsPerHash != 2 && m_openclThreadsPerHash != 4
		    && m_openclThreadsPerHash != 8) {
			cerr << "Opencl parallel hash must be 1, 2, 4, or 8.\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}

		if (m_parallelHash == 0 || m_parallelHash > 8) {
			cerr << "Cuda parallel hash must be greater than 0 and less than or equal to 8.\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}
	}

	void execute()
	{
		if (m_shouldListDevices) {
#if ETH_ETHASHCL
			if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
				CLMiner::listDevices();
#endif
#if ETH_ETHASHCUDA
			if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
				CUDAMiner::listDevices();
#endif
			exit(0);
		}

		{
			Guard l(x_log);
			loginfo << EthLime << version() << EthReset << endl << flush;
		}

		if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed) {
#if ETH_ETHASHCL
			if (m_openclDeviceCount > 0) {
				CLMiner::setDevices(m_openclDevices, m_openclDeviceCount);
				m_miningThreads = m_openclDeviceCount;
			}

			CLMiner::setCLKernel(m_openclSelectedKernel);
			CLMiner::setKernelTweak(m_openclWavetweak);
			CLMiner::setThreadsPerHash(m_openclThreadsPerHash);

			if (!CLMiner::configureGPU(
			        m_localWorkSize,
			        m_globalWorkSizeMultiplier,
			        m_openclPlatform,
			        0,
			        m_dagLoadMode,
			        m_dagCreateDevice
			    ))
				exit(1);
			CLMiner::setNumInstances(m_miningThreads);
#else
			cerr << "Selected GPU mining without having compiled with -DETHASHCL=1\n";
			exit(1);
#endif
		}
		if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed) {
#if ETH_ETHASHCUDA
			if (m_cudaDeviceCount > 0) {
				CUDAMiner::setDevices(m_cudaDevices, m_cudaDeviceCount);
				m_miningThreads = m_cudaDeviceCount;
			}

			CUDAMiner::setNumInstances(m_miningThreads);
			if (!CUDAMiner::configureGPU(
			        m_cudaBlockSize,
			        m_cudaGridSize,
			        m_numStreams,
			        m_cudaSchedule,
			        0,
			        m_dagLoadMode,
			        m_dagCreateDevice,
			        m_cudaNoEval
			    ))
				exit(1);

			CUDAMiner::setParallelHash(m_parallelHash);
#else
			cerr << "CUDA support disabled. Configure project build with -DETHASHCUDA=ON\n";
			exit(1);
#endif
		}

		map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
		sealers["opencl"] = Farm::SealerDescriptor {&CLMiner::instances, [](FarmFace & _farm, unsigned _index)
		{
			return new CLMiner(_farm, _index);
		}
		                                           };
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = Farm::SealerDescriptor {&CUDAMiner::instances, [](FarmFace & _farm, unsigned _index)
		{
			return new CUDAMiner(_farm, _index);
		}
		                                         };
#endif

		PoolClient* client = nullptr;

		client = new EthStratumClient();

		//sealers, m_minerType
		Farm f;
		f.setSealers(sealers);

		PoolManager mgr(*client, f, m_minerType);
		mgr.setReconnectTries(m_maxFarmRetries);

		mgr.addConnection(m_endpoint);

#if API_CORE
		Api api(this->m_api_port, f);
#endif

		// Start PoolManager
		mgr.start();

		// Run CLI in loop
		while (true) {
			if (mgr.isConnected()) {
				auto mp = f.miningProgress(m_show_level > 0, m_show_level > 1);
				{
					Guard l(x_log);
					loginfo << mp << f.getSolutionStats() << ' ' << f.farmLaunchedFormatted() << endl;
				}
			}
			this_thread::sleep_for(chrono::seconds(m_displayInterval));
		}
	}

private:

	/// Mining options
	MinerType m_minerType = MinerType::Mixed;
	unsigned m_openclPlatform = 0;
	unsigned m_miningThreads = UINT_MAX;
	bool m_shouldListDevices = false;
#if ETH_ETHASHCL
	unsigned m_openclSelectedKernel = 0;  ///< A numeric value for the selected OpenCL kernel
	unsigned m_openclDeviceCount = 0;
	vector<unsigned> m_openclDevices = vector<unsigned>(MAX_MINERS, -1);
	unsigned m_openclThreadsPerHash = 8;
	unsigned m_openclWavetweak = 7;
	unsigned m_globalWorkSizeMultiplier = CLMiner::c_defaultGlobalWorkSizeMultiplier;
	unsigned m_localWorkSize = CLMiner::c_defaultLocalWorkSize;
#endif
#if ETH_ETHASHCUDA
	unsigned m_cudaDeviceCount = 0;
	vector<unsigned> m_cudaDevices = vector<unsigned>(MAX_MINERS, -1);
	unsigned m_numStreams = CUDAMiner::c_defaultNumStreams;
	unsigned m_cudaSchedule = 4; // sync
	unsigned m_cudaGridSize = CUDAMiner::c_defaultGridSize;
	unsigned m_cudaBlockSize = CUDAMiner::c_defaultBlockSize;
	bool m_cudaNoEval = false;
	unsigned m_parallelHash    = 4;
#endif
	unsigned m_dagLoadMode = 0; // parallel
	unsigned m_dagCreateDevice = 0;
	/// Benchmarking params

	PoolConnection m_endpoint;
	string m_endpoint_url;

	unsigned m_maxFarmRetries = 3;
	unsigned m_farmRecheckPeriod = 500;
	unsigned m_displayInterval = 5;
	bool m_farmRecheckSet = false;
	unsigned m_show_level = 0;
#if API_CORE
	unsigned m_api_port = 0;
#endif

};

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif


int main(int argc, char** argv)
{
	// Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

	clog.imbue(std::locale(""));
	MinerCLI m;

	try {
		// Mining options:
		m.interpretOption(argc, argv);

	} catch (BadArgument ex) {
		cerr << "Error: " << ex.what() << "\n";
		exit(-1);
	}

	try {
		m.execute();
	} catch (std::exception& ex) {
		cerr << "Error: " << ex.what() << "\n";
		return 1;
	}

	return 0;
}
