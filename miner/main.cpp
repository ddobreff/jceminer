/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include <thread>
#include <fstream>
#include <iostream>
#include <miner-buildinfo.h>
#include <chrono>
#include <fstream>
#include <random>
#include <list>

#include <boost/stacktrace.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>

#include <libethcore/MinerCommon.h>
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
#include <libproto/EthStratumClient.h>
#if API_CORE
#include <libapi/api/Api.h>
#include <libapi/http/httpServer.h>
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

string version()
{
	auto* bi = miner_get_buildinfo();
	stringstream ss;
	ss << bi->project_version << "+git." << string(bi->git_commit_hash).substr(0, 7);
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
		("help,h",    bool_switch()->default_value(false), "produce help message.\n")
		("list,l",    bool_switch()->default_value(false), "List devices.\n")
		("version,v", bool_switch()->default_value(false), "list version.\n")
		("file,f",    value<string>(), "Read parameters from file.\n")
		("retries,r", value<unsigned>(&m_maxFarmRetries)->default_value(3), "Connection retries.\n")
		("email",     value<string>(&g_email), "Stratum email.\n")
		("timeout",   value<unsigned>(&g_worktimeout)->default_value(180), "Work timeout.\n")
		("hash",      bool_switch()->default_value(false), "Report hashrate to pool.\n")
		("intvl",     value<unsigned>(&m_displayInterval)->default_value(15), "statistics display interval.\n")
		("level",     value<unsigned>(&m_show_level)->default_value(0),
		 "Metrics collection level. 0 - HR only, 1 - + fan & temp, 2 - + power.\n")
		("pool,p",    value<string>(), poolDesc.str().c_str())
		("dag",       value<unsigned>(&m_dagLoadMode)->default_value(0),
		 "DAG load mode. 0 - parallel, 1 - sequential, 2 - single.\n")
		("switch",    bool_switch()->default_value(false), "Log job switch time.\n")
		("json",      bool_switch()->default_value(false), "Log formatted json messaging.\n")
		("cl,G",      bool_switch()->default_value(false), "Opencl mode.\n") // set m_minerType = MinerType::CL;
		("cu,U",      bool_switch()->default_value(false), "Cuda mode.\n") // set m_minerType = MinerType::CUDA;
		("mix,X",     bool_switch()->default_value(false),
		 "Mixed opencl and cuda mode. Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --cl-plat 1 or 2.\n")
		("eval",      bool_switch()->default_value(false),
		 "Enable software result evaluation. Use if you GPUs generate too many invalid shares.\n")
#if API_CORE
		("api",       value<unsigned>(&m_api_port)->default_value(0), "API server port number. 0 - disable, < 0 - read-only.\n")
		("http",      value<unsigned>(&m_http_port)->default_value(0), "HTTP server port number. 0 - disable\n")
#endif
#if ETH_ETHASHCL
		("cl-plat",   value<unsigned>(&m_openclPlatform)->default_value(0), "Opencl platform.\n")
		("cl-devs",   value<std::vector<unsigned>>()->multitoken(), "Opencl device list.\n")
		("cl-kern",   value<unsigned>(&m_openclSelectedKernel)->default_value(0), "Opencl kernel. 0 - Opencl, 1 - binary.\n")
#endif
#if ETH_ETHASHCUDA
		("cu-grid",   value<unsigned>(&m_cudaGridSize)->default_value(8192), "Cuda grid size.\n")
		("cu-blk",    value<unsigned>(&m_cudaBlockSize)->default_value(128), "Cuda block size.\n")
		("cu-devs",   value<std::vector<unsigned>>()->multitoken(), "Cuda device list.\n")
		("cu-para",   value<unsigned>(&m_parallelHash)->default_value(4), "Cuda parallel hashes.\n")
		("cu-sch",    value<unsigned>(&m_cudaSchedule)->default_value(4),
		 "Cuda schedule mode. 0 - auto, 1 - spin, 2 - yield, 4 - sync\n")
		("cu-strm",   value<unsigned>(&m_numStreams)->default_value(2), "Cuda streams\n")
#endif
		("stop",      value<unsigned>(&g_stopAfter)->default_value(0), "Stop after minutes. 0 - never stop.\n")
		;

		variables_map vm;

		store(parse_command_line(argc, argv, desc), vm);

		if (vm.find("file") != vm.end()) {
			// Load the file and tokenize it
			ifstream ifs(vm["file"].as<string>().c_str());
			if (!ifs) {
				cerr << "Couldn't read file " << vm["file"].as<string>() << ".\n";
				exit(-1);
			}
			// Read the whole file into a string
			stringstream ss;
			ss << ifs.rdbuf();
			// Split the file content
			boost::char_separator<char> sep(" \n\r");
			string contents(ss.str());
			boost::tokenizer<boost::char_separator<char>> tok(contents, sep);
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

		if (vm["version"].as<bool>()) {
			cout << version() << "\n";
			exit(0);
		}

		m_shouldListDevices = vm["list"].as<bool>();
		if (m_shouldListDevices)
			return;

		if (vm.count("pool") != 1) {
			cerr << "Specify a single pool URL\n";
			exit(-1);
		}

		string url = vm["pool"].as<string>();
		URI uri;
		try {
			uri = url;
		} catch (std::exception const& e) {
			cerr << "Bad endpoint address: " << url << " - " << e.what() << endl;
			exit(-1);
		}
		if (!uri.KnownScheme()) {
			cerr << "Unknown URI scheme " << uri.Scheme() << endl;
			exit(-1);
		}
		if (uri.Port() == 0) {
			cerr << "Missing port number\n";
			exit(-1);
		}
		m_endpoint = PoolConnection(uri);

		m_eval = vm["eval"].as<bool>();

#if ETH_ETHASHCUDA
		if (vm.find("cu-devs") != vm.end()) {
			m_cudaDeviceCount = vm["cu-devices"].as<vector<unsigned>>().size();
			m_cudaDevices = vm["cu-devices"].as<vector<unsigned>>();
		}

		if (m_parallelHash == 0 || m_parallelHash > 8) {
			cerr << "Cuda parallel hash must be greater than 0 and less than or equal to 8.\n";
			exit(-1);
		}
#endif

#if ETH_ETHASHCL
		if (vm.find("cl-devs") != vm.end()) {
			m_openclDeviceCount = vm["cl-devs"].as<vector<unsigned>>().size();
			m_openclDevices = vm["cl-devs"].as<vector<unsigned>>();
		}

		if (m_openclSelectedKernel > (unsigned)CLKernelName::Binary) {
			cerr << "CL kernel must be 0 or 1.\n";
			exit(-1);
		}
#endif

		if ((m_openclDeviceCount + m_cudaDeviceCount) > MAX_GPUS) {
			cerr << "Can only support up to " << MAX_GPUS << ".\n";
			exit(-1);
		}

		g_logSwitchTime = vm["switch"].as<bool>();

		g_logJson = vm["json"].as<bool>();

		g_report_stratum_hashrate = vm["hash"].as<bool>();

		if (vm["cl"].as<bool>())
			m_minerType = MinerType::CL;
		else if (vm["cu"].as<bool>())
			m_minerType = MinerType::CUDA;
		else if (vm["mix"].as<bool>())
			m_minerType = MinerType::Mixed;
		else {
			cerr << "Specify a miner type\n";
			exit(-1);
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

		if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed) {
#if ETH_ETHASHCL
			if (m_openclDeviceCount > 0) {
				CLMiner::setDevices(m_openclDevices, m_openclDeviceCount);
				m_miningThreads = m_openclDeviceCount;
			}

			CLMiner::setCLKernel(m_openclSelectedKernel);

			if (!CLMiner::configureGPU(
			        m_openclPlatform,
			        m_dagLoadMode,
			        m_dagCreateDevice,
			        m_eval
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
			        m_eval
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
		Api api(m_api_port, f);
		if (m_http_port)
			http_server.run(m_http_port, &f, &mgr);
#endif

		// Start PoolManager
		mgr.start();

		// Run CLI in loop
		while (true) {
			if (mgr.isConnected()) {
				f.collectProgress(m_show_level);
				auto mp = f.miningProgress();
				stringstream ss;
				ss << mp << '[' << f.getSolutionStats() << "] " << f.farmLaunchedFormatted();
				loginfo(ss.str());
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
	unsigned m_openclDeviceCount = 0;
	unsigned m_cudaDeviceCount = 0;
#if ETH_ETHASHCL
	unsigned m_openclSelectedKernel = 0;  ///< A numeric value for the selected OpenCL kernel
	vector<unsigned> m_openclDevices = vector<unsigned>(MAX_MINERS, -1);
	unsigned m_localWorkSize;
#endif
#if ETH_ETHASHCUDA
	vector<unsigned> m_cudaDevices = vector<unsigned>(MAX_MINERS, -1);
	unsigned m_numStreams;
	unsigned m_cudaSchedule;
	unsigned m_cudaGridSize;
	unsigned m_cudaBlockSize;
	bool m_eval = false;
	unsigned m_parallelHash    = 4;
#endif
	unsigned m_dagLoadMode = 0; // parallel
	unsigned m_dagCreateDevice = 0;
	/// Benchmarking params

	PoolConnection m_endpoint;

	unsigned m_maxFarmRetries = 3;
	unsigned m_displayInterval = 5;
	unsigned m_show_level = 0;
#if API_CORE
	unsigned m_api_port = 0;
	unsigned m_http_port = 0;
#endif

};

int main(int argc, char** argv)
{
	// Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

	clog.imbue(std::locale(""));
	MinerCLI m;

	stringstream ss;
	for (int i = 0; i < argc; i++) {
		ss << argv[i];
		if (i < argc - 1)
			ss << ' ';
	}
	loginfo(EthLime << version() << EthReset);
	loginfo("cmd: " EthWhite << ss.str() << EthReset);

	try {
		// Mining options:
		m.interpretOption(argc, argv);

	} catch (std::exception const& ex) {
		cerr << "Error parsing arguments: " << ex.what() << "\n";
		exit(-1);
	}

	try {
		m.execute();
	} catch (std::exception& ex) {
		cerr << "Error running miner: " << ex.what() << "\n";
		exit(-1);
	}

	return 0;
}
