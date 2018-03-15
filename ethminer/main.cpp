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
#include <ethminer-buildinfo.h>
#include <chrono>
#include <fstream>
#include <random>
#include <list>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/optional.hpp>

#include <libethcore/Exceptions.h>
#include <libdevcore/SHA3.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Farm.h>
#include <ethminer-buildinfo.h>
#if ETH_ETHASHCL
#include <libcl/CLMiner.h>
#endif
#if ETH_ETHASHCUDA
#include <libcuda/CUDAMiner.h>
#endif
#include <libproto/PoolManager.h>
#include <libproto/stratum/EthStratumClient.h>
#if ETH_DBUS
#include "DBusInt.h"
#endif
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

class BadArgument: public Exception {};

string version()
{
	auto* bi = ethminer_get_buildinfo();
	stringstream ss;
	ss << string("ethminer version ") + bi->project_version + "+git." + string(bi->git_commit_hash).substr(0, 7);
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

	bool interpretOption(int& i, int argc, char** argv)
	{
		string arg = argv[i];
		if (arg == "--farm-retries" && i + 1 < argc)
			m_maxFarmRetries = stol(argv[++i]);
		else if ((arg == "-SE" || arg == "--stratum-email") && i + 1 < argc)
			m_email = string(argv[++i]);
		else if ((arg == "--work-timeout") && i + 1 < argc)
			m_worktimeout = atoi(argv[++i]);
		else if ((arg == "-RH" || arg == "--report-hashrate"))
			m_report_stratum_hashrate = true;
		else if (arg == "--stats-interval" && i + 1 < argc)
			m_displayInterval = stol(argv[++i]);
		else if (arg == "-HWMON") {
			m_show_hwmonitors = true;
			if ((i + 1 < argc) && (*argv[i + 1] != '-'))
				m_show_power = (bool)atoi(argv[++i]);
		} else if ((arg == "-P") && (i + 1 < argc)) {
			if (m_endpoint.Empty()) {
				string url = argv[++i];
				if (url == "exit") // add fake scheme and port to 'exit' url
					url = "stratum://exit:1";
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
			} else {
				cerr << "Specify only one pool\n";
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
#if API_CORE
		else if ((arg == "--api-port") && i + 1 < argc)
			m_api_port = atoi(argv[++i]);
#endif
#if ETH_ETHASHCL
		else if (arg == "--cl-platform" && i + 1 < argc)
			m_openclPlatform = stol(argv[++i]);
		else if (arg == "--cl-devices" || arg == "--cl-device")
			while (m_openclDeviceCount < MAX_MINERS && i + 1 < argc) {
				m_openclDevices[m_openclDeviceCount] = stol(argv[++i]);
				++m_openclDeviceCount;
			}
		else if (arg == "--cl-parallel-hash" && i + 1 < argc) {
			m_openclThreadsPerHash = stol(argv[++i]);
			if (m_openclThreadsPerHash != 1 && m_openclThreadsPerHash != 2 && m_openclThreadsPerHash != 4
			    && m_openclThreadsPerHash != 8)
				BOOST_THROW_EXCEPTION(BadArgument());
		} else if (arg == "--cl-kernel" && i + 1 < argc)
			m_openclSelectedKernel = stol(argv[++i]);
		else if (arg == "--cl-wavetweak" && i + 1 < argc)
			m_openclWavetweak = stol(argv[++i]);
		else if (arg == "--cl-global-work"  && i + 1 < argc) {
			i++;
			m_globalWorkSizeMultiplier =
			    (strcmp(argv[i], "auto") == 0) ? 0 : stol(argv[i]);
		} else if (arg == "--cl-local-work" && i + 1 < argc)
			m_localWorkSize = stol(argv[++i]);
#endif
#if ETH_ETHASHCL || ETH_ETHASHCUDA
		else if ((arg == "-l") || (arg == "--list-devices"))
			m_shouldListDevices = true;
#endif
#if ETH_ETHASHCUDA
		else if (arg == "--cu-grid-size" && i + 1 < argc)
			m_cudaGridSize = stol(argv[++i]);
		else if (arg == "--cu-block-size" && i + 1 < argc)
			m_cudaBlockSize = stol(argv[++i]);
		else if (arg == "--cu-devices") {
			while (m_cudaDeviceCount < MAX_MINERS && i + 1 < argc) {
				m_cudaDevices[m_cudaDeviceCount] = stol(argv[++i]);
				++m_cudaDeviceCount;
			}
		} else if (arg == "--cu-parallel-hash" && i + 1 < argc) {
			m_parallelHash = stol(argv[++i]);
			if (m_parallelHash == 0 || m_parallelHash > 8)
				throw BadArgument();
		} else if (arg == "--cu-schedule" && i + 1 < argc) {
			string mode = argv[++i];
			if (mode == "auto") m_cudaSchedule = 0;
			else if (mode == "spin") m_cudaSchedule = 1;
			else if (mode == "yield") m_cudaSchedule = 2;
			else if (mode == "sync") m_cudaSchedule = 4;
			else {
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		} else if (arg == "--cu-streams" && i + 1 < argc)
			m_numStreams = stol(argv[++i]);
		else if (arg == "--cu-noeval")
			m_cudaNoEval = true;
#endif
		else if ((arg == "-L" || arg == "--dag-load-mode") && i + 1 < argc) {
			string mode = argv[++i];
			if (mode == "parallel") m_dagLoadMode = DAG_LOAD_MODE_PARALLEL;
			else if (mode == "sequential") m_dagLoadMode = DAG_LOAD_MODE_SEQUENTIAL;
			else if (mode == "single") {
				m_dagLoadMode = DAG_LOAD_MODE_SINGLE;
				m_dagCreateDevice = stol(argv[++i]);
			} else {
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		} else if (arg == "--benchmark-warmup" && i + 1 < argc)
			m_benchmarkWarmup = stol(argv[++i]);
		else if (arg == "--benchmark-trial" && i + 1 < argc)
			m_benchmarkTrial = stol(argv[++i]);
		else if (arg == "--benchmark-trials" && i + 1 < argc)
			m_benchmarkTrials = stol(argv[++i]);
		else if (arg == "--log-switch")
			g_logSwitchTime = true;
		else if (arg == "--log-json")
			g_logJson = true;
		else if (arg == "-G" || arg == "--cl")
			m_minerType = MinerType::CL;
		else if (arg == "-U" || arg == "--cua")
			m_minerType = MinerType::CUDA;
		else if (arg == "-X" || arg == "--cu-cl")
			m_minerType = MinerType::Mixed;
		else
			return false;
		return true;
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
			loginfo << version() << endl;
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

		client = new EthStratumClient(m_worktimeout, m_email, m_report_stratum_hashrate);

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
				auto mp = f.miningProgress(m_show_hwmonitors, m_show_power);
				{
					Guard l(x_log);
					loginfo << mp << f.getSolutionStats() << ' ' << f.farmLaunchedFormatted() << endl;
				}
#if ETH_DBUS
				dbusint.send(toString(mp).data());
#endif
			}
			this_thread::sleep_for(chrono::seconds(m_displayInterval));
		}
	}

	static void streamHelp(ostream& _out)
	{
		_out << " Mining configuration:\n"
		     "    -G,--cl  When mining use the GPU via OpenCL.\n"
		     "    -U,--cu  When mining use the GPU via CUDA.\n"
		     "    -X,--cu-cl Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --cl-platform 1 or 2.\n"
		     "        Use --list-devices option to check which platform is your AMD.\n"
		     "    -l, --list-devices List the detected OpenCL/CUDA devices and exit. Should be combined with -G or -U flag\n"
		     "    --stats-interval <n> Set mining stats display interval in seconds. (default: every 5 seconds)\n"
		     "    -L, --dag-load-mode <mode> DAG generation mode.\n"
		     "        parallel    - load DAG on all GPUs at the same time (default)\n"
		     "        sequential  - load DAG on GPUs one after another. Use this when the miner crashes during DAG generation\n"
		     "        single <n>  - generate DAG on device n, then copy to other devices\n"
		     "    --work-timeout <n> reconnect/failover after n seconds of working on the same job. Defaults to 180. Don't set lower\n"
		     "        than max. avg. block time\n"
		     "    -RH, --report-hashrate Report current hashrate to pool (please only enable on pools supporting this)\n"
		     "    -HWMON [<n>], Displays gpu temp, fan percent and power usage. Note: In linux, the program uses sysfs, which may\n"
		     "        require running with root priviledges.\n"
		     "        0: Displays only temp and fan percent (default)\n"
		     "        1: Also displays power usage\n"
		     "    --exit Stops the miner whenever an error is encountered\n"
		     "    -SE, --stratum-email <s> Email address used in eth-proxy (optional)\n"
		     "    -P URL Specify a pool URL. Can be used multiple times. The 1st for for the primary pool, and the 2nd for the failover pool.\n"
		     "        URL takes the form: scheme://[user[:password]@]hostname:port.\n"
		     "          unsecured schemes:    " << URI::KnownSchemes(SecureLevel::NONE) << endl;
		_out << "          secured with any TLS: " << URI::KnownSchemes(SecureLevel::TLS) << endl;
		_out << "          secured with TLS 1.2: " << URI::KnownSchemes(SecureLevel::TLS12) << endl;
		_out << "        Example: stratum+ssl://0x012345678901234567890234567890123.miner1@ethermine.org:5555\n"
		     "        Stratum versions:\n"
		     "          stratum  - official stratum spec: ethpool, ethermine, coinotron, mph, nanopool\n"
		     "          stratum1 - eth-proxy compatible: dwarfpool, f2pool, nanopool (required for hashrate reporting to work with nanopool)\n"
		     "          stratum2 - EthereumStratum/1.0.0: nicehash\n\n";
#if ETH_ETHASHCL
		_out << " OpenCL configuration:\n"
		     "    --cl-platform <n>  When mining using -G/--cl use OpenCL platform n (default: 0).\n"
		     "    --cl-device <n>  When mining using -G/--cl use OpenCL device n (default: 0).\n"
		     "    --cl-devices <0 1 ..n> Select which OpenCL devices to mine on. Default is to use all\n"
		     "    --cl-kernel <n>  Use a different OpenCL kernel (default: use stable kernel)\n"
		     "        0: stable kernel\n"
		     "        1: experimental kernel\n"
		     "        2: binary kernel\n";
		_out << "    --cl-local-work Set the OpenCL local work size. Default is " << CLMiner::c_defaultLocalWorkSize << endl
		     << "    --cl-global-work Set the OpenCL global work size as a multiple of the local work size. Default is "
		     << CLMiner::c_defaultGlobalWorkSizeMultiplier << " * " << CLMiner::c_defaultLocalWorkSize << endl;
		_out << "        You may also specify auto for optimal Radeon value based on configuration.\n"
		     "    --cl-parallel-hash <1 2 ..8> Define how many threads to associate per hash. Default=8\n"
		     "    --cl-wavetweak 0-100 \n\n";
#endif
#if ETH_ETHASHCUDA
		_out << " CUDA configuration:\n"
		     "    --cu-block-size Set the CUDA block work size. Default is " << toString(CUDAMiner::c_defaultBlockSize) << endl
		     << "    --cu-grid-size Set the CUDA grid size. Default is " << toString(CUDAMiner::c_defaultGridSize) << endl
		     << "    --cu-streams Set the number of CUDA streams. Default is " << toString(CUDAMiner::c_defaultNumStreams) << endl
		     << "    --cu-schedule <mode> Set the schedule mode for CUDA threads waiting for CUDA devices to finish work.\n"
		     "         Default is 'sync'. Possible values are:\n"
		     "        auto  - Uses a heuristic based on the number of active CUDA contexts in the process C and the number of logical\n"
		     "            processors in the system P. If C > P, then yield else spin.\n"
		     "        spin  - Instruct CUDA to actively spin when waiting for results from the device.\n"
		     "        yield - Instruct CUDA to yield its thread when waiting for results from the device.\n"
		     "        sync  - Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the results\n"
		     "            from the device.\n"
		     "    --cu-devices <0 1 ..n> Select which CUDA GPUs to mine on. Default is to use all\n"
		     "    --cu-parallel-hash <1 2 ..8> Define how many hashes to calculate in a kernel, can be scaled to achieve better\n"
		     "            performance. Default=4\n"
		     "    --cu-noeval  bypass host software re-evalution of GPU solutions.\n"
		     "        This will trim some milliseconds off the time it takes to send a result to the pool.\n"
		     "        Use at your own risk! If GPU generates errored results they WILL be forwarded to the pool\n"
		     "        Not recommended at high overclock.\n\n";
#endif
#if API_CORE
		_out << " API core configuration:\n"
		     "    --api-port Set the api port, the miner should listen to. Use 0 to disable. Default=0, use negative numbers to\n"
		     "         run in readonly mode. for example -3333.\n\n"
		     " Logging:\n"
		     "    --log-switch Display per card switch time.\n"
		     "    --log-json Display formatted json I/O.\n\n";
#endif
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
	unsigned m_benchmarkWarmup = 15;
	unsigned m_benchmarkTrial = 3;
	unsigned m_benchmarkTrials = 5;
	unsigned m_benchmarkBlock = 0;

	PoolConnection m_endpoint;

	unsigned m_maxFarmRetries = 3;
	unsigned m_farmRecheckPeriod = 500;
	unsigned m_displayInterval = 5;
	bool m_farmRecheckSet = false;
	int m_worktimeout = 180;
	bool m_show_hwmonitors = false;
	bool m_show_power = false;
#if API_CORE
	int m_api_port = 0;
#endif

	bool m_report_stratum_hashrate = false;
	string m_email;

#if ETH_DBUS
	DBusInt dbusint;
#endif
};

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif


void help()
{
	cout << "Usage ethminer [OPTIONS]\n"
	     "Options:\n\n";
	MinerCLI::streamHelp(cout);
	cout << " General Options:\n"
	     "\n"
	     "    -V,--version  Show the version and exit.\n"
	     "    -h,--help  Show this help message and exit.\n\n";
	exit(0);
}

int main(int argc, char** argv)
{
	// Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

	clog.imbue(std::locale(""));
	MinerCLI m;

	try {
		for (int i = 1; i < argc; ++i) {
			// Mining options:
			if (m.interpretOption(i, argc, argv))
				continue;

			// Standard options:
			string arg = argv[i];
			if (arg == "-h" || arg == "--help")
				help();
			else if (arg == "-V" || arg == "--version") {
				cout << version();
				exit(0);
			} else {
				cerr << "Invalid argument: " << arg << endl;
				exit(-1);
			}
		}
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
