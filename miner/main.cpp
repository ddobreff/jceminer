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
#include <boost/tokenizer.hpp>

#include <libethcore/MinerCommon.h>
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
unsigned g_stopAfter;

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
		using namespace boost::program_options;

		stringstream poolDesc;
		poolDesc
		        << "URL takes the form:\nscheme://[user[:password]@]hostname:port\n\n"
		        << "unsecured schemes: " << URI::KnownSchemes(SecureLevel::NONE) << '\n'
		        << "secured with any TLS: " << URI::KnownSchemes(SecureLevel::TLS) << '\n'
		        << "secured with TLS 1.2: " << URI::KnownSchemes(SecureLevel::TLS12) << "\n\n"
		        << "Example:\n"
		        << "stratum+tls://0x0123456...7890190123.miner1@ethermine.org:5555\n\n"
		        << "Stratum versions:\n"
		        << "stratum  - ethpool, ethermine, coinotron, mph, nanopool\n"
		        << "eth-proxy - dwarfpool, f2pool, nanopool (required for hashrate reporting to work with nanopool)\n"
		        << "nicehash - Ethereum Stratum\n";

		options_description desc("Options");
		desc.add_options()
		("help,h", bool_switch()->default_value(false), "produce help message.\n")
		("devs,d", bool_switch()->default_value(false), "List devices.\n")
		("vers,v", bool_switch()->default_value(false), "list version.\n")
		("file,f", value<string>(), "Read parameters from file.\n")
		("retries,r", value<unsigned>(&m_maxFarmRetries)->default_value(3), "Connection retries.\n")
		("email,e", value<string>(&g_email), "Stratum email.\n")
		("timeout,w", value<unsigned>(&g_worktimeout)->default_value(180), "Work timeout.\n")
		("hash", bool_switch()->default_value(false), "Report hashrate to pool.\n")
		("s-intvl,s", value<unsigned>(&m_displayInterval)->default_value(15), "statistics display interval.\n")
		("s-level,l", value<unsigned>(&m_show_level)->default_value(0),
		 "statistics display interval. 0 - HR only, 1 - + fan & temp, 2 - + power.\n")
		("pool,p", value<string>(&m_endpoint_url), poolDesc.str().c_str())
		("dag", value<unsigned>(&m_dagLoadMode)->default_value(0),
		 "DAG load mode. 0 - parallel, 1 - sequential, 2 - single.\n")
		("l-switch", bool_switch()->default_value(false), "Log job switch time.\n")
		("l-json", bool_switch()->default_value(false), "Log formatted json messaging.\n")
		("cl,G", bool_switch()->default_value(false), "Opencl mode.\n") // set m_minerType = MinerType::CL;
		("cu,U", bool_switch()->default_value(false), "Cuda mode.\n") // set m_minerType = MinerType::CUDA;
		("mix,X", bool_switch()->default_value(false),
		 "Mixed opencl and cuda mode. Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --cl-platform 1 or 2.\n")
#if API_CORE
		("api-port,a", value<unsigned>(&m_api_port)->default_value(0), "API port number. 0 - disable, < 0 - read-only.\n")
#endif
#if ETH_ETHASHCL
		("cl-plat", value<unsigned>(&m_openclPlatform)->default_value(0), "Opencl platform.\n")
		("cl-devs", value<std::vector<unsigned>>()->multitoken(), "Opencl device list.\n")
		("cl-para", value<unsigned>(&m_openclThreadsPerHash)->default_value(8), "Opencl parallel hashes.\n")
		("cl-kern", value<unsigned>(&m_openclSelectedKernel)->default_value(1),
		 "Opencl kernel. 0 - Stable, 1 - Experimental, 2 - binary.\n")
		("cl-tweak", value<unsigned>(&m_openclWavetweak)->default_value(7), "Opencl wave tweak.\n")
		("cl-global", value<unsigned>(&m_globalWorkSizeMultiplier)->default_value(8192), "Opencl global work size. 0 - Auto.\n")
		("cl-local", value<unsigned>(&m_localWorkSize)->default_value(64), "Opencl local work size.\n")
#endif
#if ETH_ETHASHCUDA
		("cu-grid", value<unsigned>(&m_cudaGridSize)->default_value(8192), "Cuda grid size.\n")
		("cu-block", value<unsigned>(&m_cudaBlockSize)->default_value(128), "Cuda block size.\n")
		("cu-devs", value<std::vector<unsigned>>()->multitoken(), "Cuda device list.\n")
		("cu-para", value<unsigned>(&m_parallelHash)->default_value(4), "Cuda parallel hashes.\n")
		("cu-sch", value<unsigned>(&m_cudaSchedule)->default_value(4),
		 "Cuda schedule mode. 0 - auto, 1 - spin, 2 - yield, 4 - sync\n")
		("cu-stream", value<unsigned>(&m_numStreams)->default_value(2), "Cuda streams\n")
		("cu-noeval", bool_switch()->default_value(false), "Cuda bypass software result evaluation.\n")
#endif
		("stop", value<unsigned>(&g_stopAfter)->default_value(0), "Stop after seconds\n")
		;

		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);

		if (vm.find("file") != vm.end()) {
			// Load the file and tokenize it
			ifstream ifs(vm["file"].as<string>().c_str());
			if (!ifs) {
				cerr << "Couldn't read file " << vm["file"].as<string>() << ".\n";
				BOOST_THROW_EXCEPTION(BadArgument());
			}
			// Read the whole file into a string
			stringstream ss;
			ss << ifs.rdbuf();
			// Split the file content
			boost::char_separator<char> sep(" \n\r");
			string ResponsefileContents(ss.str());
			boost::tokenizer<boost::char_separator<char>> tok(ResponsefileContents, sep);
			vector<string> args;
			copy(tok.begin(), tok.end(), back_inserter(args));
			// Parse the file and store the options
			store(command_line_parser(args).options(desc).run(), vm);
		}

		notify(vm);
		if (vm["help"].as<bool>()) {
			cout << desc << "\n";
			exit(0);
		}

		if (vm["vers"].as<bool>()) {
			cout << version() << "\n";
			exit(0);
		}

		if (vm["devs"].as<bool>()) {
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

#if ETH_ETHASHCUDA
		if (vm.find("cu-devs") != vm.end()) {
			m_cudaDeviceCount = vm["cu-devices"].as<vector<unsigned>>().size();
			m_cudaDevices = vm["cu-devices"].as<vector<unsigned>>();
		}

		if (m_parallelHash == 0 || m_parallelHash > 8) {
			cerr << "Cuda parallel hash must be greater than 0 and less than or equal to 8.\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}

		m_cudaNoEval = vm["cu-noeval"].as<bool>();
#endif

#if ETH_ETHASHCL
		if (vm.find("cl-devs") != vm.end()) {
			m_openclDeviceCount = vm["cl-devices"].as<vector<unsigned>>().size();
			m_openclDevices = vm["cl-devices"].as<vector<unsigned>>();
		}

		if (m_openclThreadsPerHash != 1 && m_openclThreadsPerHash != 2 && m_openclThreadsPerHash != 4
		    && m_openclThreadsPerHash != 8) {
			cerr << "Opencl parallel hash must be 1, 2, 4, or 8.\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}
#endif

#if ETH_ETHASHCL && ETH_ETHASHCUDA
		if ((m_openclDeviceCount + m_cudaDeviceCount) > MAX_GPUS) {
			cerr << "Can only support up to " << MAX_GPUS << ".\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}
#endif
#if ETH_ETHASHCL && !ETH_ETHASHCUDA
		if (m_openclDeviceCount > MAX_GPUS) {
			cerr << "Can only support up to " << MAX_GPUS << ".\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}
#endif
#if !ETH_ETHASHCL && ETH_ETHASHCUDA
		if (m_cudaDeviceCount > MAX_GPUS) {
			cerr << "Can only support up to " << MAX_GPUS << ".\n";
			BOOST_THROW_EXCEPTION(BadArgument());
		}
#endif

		g_logSwitchTime = vm["l-switch"].as<bool>();

		g_logJson = vm["l-json"].as<bool>();

		g_report_stratum_hashrate = vm["hash"].as<bool>();

		if (vm["cl"].as<bool>())
			m_minerType = MinerType::CL;
		else if (vm["cu"].as<bool>())
			m_minerType = MinerType::CUDA;
		else if (vm["mix"].as<bool>())
			m_minerType = MinerType::Mixed;
		else {
			cerr << "Specify a miner type\n";
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
	unsigned m_globalWorkSizeMultiplier;
	unsigned m_localWorkSize;
#endif
#if ETH_ETHASHCUDA
	unsigned m_cudaDeviceCount = 0;
	vector<unsigned> m_cudaDevices = vector<unsigned>(MAX_MINERS, -1);
	unsigned m_numStreams;
	unsigned m_cudaSchedule;
	unsigned m_cudaGridSize;
	unsigned m_cudaBlockSize;
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
