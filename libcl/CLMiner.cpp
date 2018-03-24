/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "CLMiner.h"
#include "libethash/internal.h"
#include "libdevcore/Log.h"
#include "CLMiner_kernel.h"
#include <boost/dll.hpp>

using namespace dev;
using namespace eth;

typedef struct {
	unsigned workGroupSize;
	unsigned workMultiplier;
	unsigned workThreads;
	unsigned workTweak;
} clConfig;

std::map <std::string, clConfig> optimalConfigs = {
//                      group   mult    threads tweak
	{"opencl",          {256,   8192,  2,      0}},
	{"ellesmere",       {64,    32768,  8,      7}}
};

namespace dev
{
namespace eth
{

typedef struct {
	uint32_t count;
	uint32_t gid;
	uint64_t mix[4];
} search_results;


CLKernelName CLMiner::s_clKernelName;

namespace
{

void addDefinition(string& _source, char const* _id, unsigned _value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", _id, _value);
	_source.insert(_source.begin(), buf, buf + strlen(buf));
}

std::vector<cl::Platform> getPlatforms()
{
	vector<cl::Platform> platforms;
	try {
		cl::Platform::get(&platforms);
	} catch (std::exception const& err) {
		throw err;
	}
	return platforms;
}

std::vector<cl::Device> getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
	vector<cl::Device> devices;
	size_t platform_num = min<size_t>(_platformId, _platforms.size() - 1);
	try {
		_platforms[platform_num].getDevices(
		    CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
		    &devices
		);
	} catch (std::exception const&) {
		// simply no devices found return empty vector
	}
	return devices;
}

}

}
}

bool CLMiner::s_eval = false;
unsigned CLMiner::s_platformId = 0;
unsigned CLMiner::s_numInstances = 0;
vector<int> CLMiner::s_devices(MAX_MINERS, -1);

CLMiner::CLMiner(FarmFace& _farm, unsigned _index):
	Miner("cl-", _farm, _index)
{
}

CLMiner::~CLMiner()
{
}

void CLMiner::workLoop()
{
	// Memory for zero-ing buffers. Cannot be static because crashes on macOS.
	uint32_t const c_zero = 0;

	uint64_t startNonce = 0;

	// The work package currently processed by GPU.
	WorkPackage current;
	current.header = h256{1u};
	current.seed = h256{1u};

	try {
		while (true) {
			const WorkPackage w = work();

			if (current.header != w.header) {
				// New work received. Update GPU data.
				if (!w) {
					{
						Guard l(x_log);
						logwarn << workerName() << " - No work. Pause for 3 s." << endl;
					}
					std::this_thread::sleep_for(std::chrono::seconds(3));
					continue;
				}

				if (current.seed != w.seed) {
					if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL) {
						while (s_dagLoadIndex < index)
							this_thread::sleep_for(chrono::seconds(1));
						++s_dagLoadIndex;
					}

					{
						Guard l(x_log);
						loginfo << workerName() << " - New seed " << w.seed << endl;
					}
					init(w.seed);
				}

				// Upper 64 bits of the boundary.
				const uint64_t target = (uint64_t)(u64)((u256)w.boundary >> 192);
				assert(target > 0);

				// Update header constant buffer.
				m_queue.enqueueWriteBuffer(m_header, CL_FALSE, 0, w.header.size, w.header.data());
				m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(c_zero), &c_zero);
				m_searchKernel.setArg(0, m_searchBuffer);  // Supply output buffer to kernel.
				m_searchKernel.setArg(4, target);

				if (w.exSizeBits >= 0) {
					// This can support up to 2^c_log2MaxMiners devices.
					startNonce = w.startNonce | ((uint64_t)index << (64 - LOG2_MAX_MINERS - w.exSizeBits));
				} else
					startNonce = get_start_nonce();

				if (g_logSwitchTime) {
					Guard l(x_log);
					loginfo << workerName() << " - switch time "
					        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
					                workSwitchStart).count() << " ms." << endl;
				}
			}

			// Read results.
			// TODO: could use pinned host pointer instead.
			search_results results;
			m_queue.enqueueReadBuffer(m_searchBuffer, CL_TRUE, 0, sizeof(results), &results);

			if (results.count) {
				// Reset search buffer if any solution found.
				m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(c_zero), &c_zero);
			}

			// Run the kernel.
			m_searchKernel.setArg(3, startNonce);
			m_queue.enqueueNDRangeKernel(m_searchKernel, cl::NullRange, m_workMultiplier * m_workgroupSize, m_workgroupSize);

			// Report results while the kernel is running.
			if (results.count) {
				uint64_t nonce = current.startNonce + results.gid;
				if (s_eval || (s_clKernelName != CLKernelName::Opencl)) {
					Result r = EthashAux::eval(current.seed, current.header, nonce);
					if (r.value < current.boundary)
						farm.submitProof(workerName(), Solution{nonce, r.mixHash, current, current.header != w.header});
					else {
						farm.failedSolution();
						{
							Guard l(x_log);
							logerror << workerName() << " - discarded incorrect result!\n";
						}
					}
				} else {
					h256 mix;
					memcpy(mix.data(), results.mix, sizeof(results.mix));
					farm.submitProof(workerName(), Solution{nonce, mix, current, current.header != w.header});
				}
			}

			current = w;        // kernel now processing newest work
			current.startNonce = startNonce;
			// Increase start nonce for following kernel execution.
			startNonce += m_workMultiplier * m_workgroupSize;

			// Report hash count
			addHashCount(m_workMultiplier * m_workgroupSize);

			// Make sure the last buffer write has finished --
			// it reads local variable.
			m_queue.finish();
		}

	} catch (std::exception const& _e) {
		logerror << workerName() << " - " << _e.what() << endl;
		throw;
	}
}

void CLMiner::kick_miner() {}

unsigned CLMiner::getNumDevices()
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return 0;

	vector<cl::Device> devices = getDevices(platforms, s_platformId);
	if (devices.empty()) {
		{
			Guard l(x_log);
			logerror << "No OpenCL devices found." << endl;
		}
		return 0;
	}
	return devices.size();
}

void CLMiner::listDevices()
{
	string outString = "\nListing OpenCL devices.\nFORMAT: [platformID] [deviceID] device name\n";
	unsigned int i = 0;

	try {
		vector<cl::Platform> platforms = getPlatforms();
		if (platforms.empty())
			return;
		for (unsigned j = 0; j < platforms.size(); ++j) {
			i = 0;
			vector<cl::Device> devices = getDevices(platforms, j);
			for (auto const& device : devices) {
				outString += "[" + to_string(j) + "] [" + to_string(i) + "] " + device.getInfo<CL_DEVICE_NAME>() + "\n";
				outString += "\tCL_DEVICE_TYPE: ";
				switch (device.getInfo<CL_DEVICE_TYPE>()) {
				case CL_DEVICE_TYPE_CPU:
					outString += "CPU\n";
					break;
				case CL_DEVICE_TYPE_GPU: {
					cl_uint maxCus;
					clGetDeviceInfo(device(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxCus), &maxCus, NULL);
					stringstream ss;
					ss << maxCus;
					outString += "GPU #CUs = " + ss.str() + '\n';
				}
				break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					outString += "ACCELERATOR\n";
					break;
				default:
					outString += "DEFAULT\n";
					break;
				}
				outString += "\tCL_DEVICE_GLOBAL_MEM_SIZE: " + to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) + "\n";
				outString += "\tCL_DEVICE_MAX_MEM_ALLOC_SIZE: " + to_string(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) + "\n";
				outString += "\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " + to_string(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
				++i;
			}
		}
		std::cout << outString;
	} catch (std::exception const&) {
	}
}

bool CLMiner::configureGPU(
    unsigned _platformId,
    unsigned _dagLoadMode,
    unsigned _dagCreateDevice,
    bool _eval
)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;
	s_eval = _eval;

	s_platformId = _platformId;

	uint64_t dagSize = ethash_get_datasize(0);

	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return false;
	if (_platformId >= platforms.size())
		return false;

	vector<cl::Device> devices = getDevices(platforms, _platformId);
	for (auto const& device : devices) {
		cl_ulong result = 0;
		device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
		if (result >= dagSize) {
			{
				Guard l(x_log);
				loginfo << "Found suitable OpenCL device [" << device.getInfo<CL_DEVICE_NAME>() << "] with " << result <<
				        " bytes of GPU memory" << endl;
			}
			return true;
		}

		{
			Guard l(x_log);
			logerror << "OpenCL device " << device.getInfo<CL_DEVICE_NAME>() << " has insufficient GPU memory." << result <<
			         " bytes of memory found < " << dagSize << " bytes of memory required" << endl;
		}
	}

	cout << "No GPU device with sufficient memory was found. Can't GPU mine. Remove the -G argument" << endl;
	return false;
}

bool CLMiner::init(const h256& seed)
{
	EthashAux::LightType light = EthashAux::light(seed);

	// get all platforms
	try {
		vector<cl::Platform> platforms = getPlatforms();
		if (platforms.empty())
			return false;

		// use selected platform
		unsigned platformIdx = min<unsigned>(s_platformId, platforms.size() - 1);

		string platformName = platforms[platformIdx].getInfo<CL_PLATFORM_NAME>();
		{
			Guard l(x_log);
			loginfo << workerName() << " - Platform: " << platformName << endl;
		}

		int platformId = OPENCL_PLATFORM_UNKNOWN;
		{
			// this mutex prevents race conditions when calling the adl wrapper since it is apparently not thread safe
			static std::mutex mtx;
			std::lock_guard<std::mutex> lock(mtx);

			if (platformName == "NVIDIA CUDA") {
				platformId = OPENCL_PLATFORM_NVIDIA;
				m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
				m_hwmoninfo.indexSource = HwMonitorIndexSource::OPENCL;
			} else if (platformName == "AMD Accelerated Parallel Processing") {
				platformId = OPENCL_PLATFORM_AMD;
				m_hwmoninfo.deviceType = HwMonitorInfoType::AMD;
				m_hwmoninfo.indexSource = HwMonitorIndexSource::OPENCL;
			} else if (platformName == "Clover")
				platformId = OPENCL_PLATFORM_CLOVER;
		}

		// get GPU device of the default platform
		vector<cl::Device> devices = getDevices(platforms, platformIdx);
		if (devices.empty()) {
			{
				Guard l(x_log);
				logerror << workerName() << " - No OpenCL devices found." << endl;
			}
			return false;
		}

		// use selected device
		int idx = index % devices.size();
		unsigned deviceId = s_devices[idx] > -1 ? s_devices[idx] : index;
		m_hwmoninfo.deviceIndex = deviceId % devices.size();
		cl::Device& device = devices[deviceId % devices.size()];
		string device_version = device.getInfo<CL_DEVICE_VERSION>();
		{
			Guard l(x_log);
			loginfo << workerName() << " - Device: " << device.getInfo<CL_DEVICE_NAME>() << " / " << device_version << endl;
		}

		string clVer = device_version.substr(7, 3);
		if (clVer == "1.0" || clVer == "1.1") {
			if (platformId == OPENCL_PLATFORM_CLOVER) {
				Guard l(x_log);
				logwarn << workerName() << " - OpenCL " << clVer <<
				        " not supported, but platform Clover might work nevertheless. USE AT OWN RISK!" <<
				        endl;
			} else {
				{
					Guard l(x_log);
					logerror << workerName() << " - OpenCL " << clVer << " not supported - minimum required version is 1.2" << endl;
				}
				return false;
			}
		}

		char options[256];
		int computeCapability = 0;
		if (platformId == OPENCL_PLATFORM_NVIDIA) {
			cl_uint computeCapabilityMajor;
			cl_uint computeCapabilityMinor;
			clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &computeCapabilityMajor, NULL);
			clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &computeCapabilityMinor, NULL);

			computeCapability = computeCapabilityMajor * 10 + computeCapabilityMinor;
			int maxregs = computeCapability >= 35 ? 72 : 63;
			sprintf(options, "-cl-nv-maxrregcount=%d", maxregs);
		} else
			sprintf(options, "%s", "");
		// create context
		m_context = cl::Context(vector<cl::Device>(&device, &device + 1));
		m_queue = cl::CommandQueue(m_context, device);

		uint64_t dagSize = ethash_get_datasize(light->light->block_number);
		uint32_t dagSize128 = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(light->data().size() / sizeof(node));

		// patch source code
		// note: The kernels here are simply compiled version of the respective .cl kernels
		// into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
		// See libethash-cl/CMakeLists.txt: add_custom_command()
		// TODO: Just use C++ raw string literal.
		string code;

		if (s_clKernelName == CLKernelName::Opencl) {
			{
				Guard l(x_log);
				loginfo << workerName() << " - OpenCL kernel: opencl kernel" << endl;
			}
			code = string(CLMiner_kernel, CLMiner_kernel + sizeof(CLMiner_kernel));
		} else { // Fallback to experimental kernel if binary loader fails
			{
				Guard l(x_log);
				loginfo << workerName() << " - OpenCL kernel: " << (s_clKernelName == CLKernelName::Binary ?  "Binary" : "opencl") <<
				        " kernel" <<
				        endl;
			}
			code = string(CLMiner_kernel, CLMiner_kernel + sizeof(CLMiner_kernel));
		}

		clConfig conf;
		if (s_clKernelName == CLKernelName::Opencl)
			conf = optimalConfigs["opencl"];
		else { /* if (s_clKernelName == CLKernelName::Binary) */
			std::string name = device.getInfo<CL_DEVICE_NAME>();
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);
			if (optimalConfigs.find(name) == optimalConfigs.end()) {
				logerror << workerName() << " - Can't find configuration for binary kernel " << name << endl;
				throw runtime_error("No kernel");
			}
			conf = optimalConfigs[name];
		}
		m_threadsPerHash = conf.workThreads;
		m_workgroupSize = conf.workGroupSize;
		m_workMultiplier = conf.workMultiplier;
		m_threadTweak = conf.workTweak;

		unsigned int computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		// Apparently some 36 CU devices return a bogus 14!!!
		computeUnits = computeUnits == 14 ? 36 : computeUnits;
		logwarn << workerName() << " - adjusting CL work multiplier for " << computeUnits << " CUs.\n";
		m_workMultiplier = (m_workMultiplier * computeUnits) / 36;
		logwarn << workerName() << " - threads per hash " << m_threadsPerHash
		        << ", work group " << m_workgroupSize
		        << ", work multiplier " << m_workMultiplier
		        << ", work tweak " << m_threadTweak << endl;

		addDefinition(code, "GROUP_SIZE", m_workgroupSize);
		addDefinition(code, "DAG_SIZE", dagSize128);
		addDefinition(code, "LIGHT_SIZE", lightSize64);
		addDefinition(code, "ACCESSES", ETHASH_ACCESSES);
		addDefinition(code, "PLATFORM", platformId);
		addDefinition(code, "COMPUTE", computeCapability);
		addDefinition(code, "THREADS_PER_HASH", m_threadsPerHash);

		// create miner OpenCL program
		cl::Program::Sources sources{{code.data(), code.size()}};
		cl::Program program(m_context, sources), binaryProgram;
		try {
			program.build({device}, options);
		} catch (std::exception const&) {
			logerror << workerName() << " - Build info: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
			return false;
		}

		// If we have a binary kernel, we load it in tandem with the opencl,
		// that way, we can use the dag generate opencl code
		bool loadedBinary = false;

		if (s_clKernelName >= CLKernelName::Binary) {
			std::ifstream kernel_file;
			vector<unsigned char> bin_data;
			std::stringstream fname_strm;

			/* Open kernels/{device.getInfo<CL_DEVICE>}.bin */
			std::string name = device.getInfo<CL_DEVICE_NAME>();
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);
			fname_strm << boost::dll::program_location().parent_path().string() << "/kernels/" << name << ".bin";

			kernel_file.open(
			    fname_strm.str(),
			    ios::in | ios::binary
			);

			if (kernel_file.good()) {

				/* Load the data vector with file data */
				kernel_file.unsetf(std::ios::skipws);
				bin_data.insert(bin_data.begin(),
				                std::istream_iterator<unsigned char>(kernel_file),
				                std::istream_iterator<unsigned char>());

				/* Setup the program */
				cl::Program::Binaries blobs({bin_data});
				cl::Program program(m_context, { device }, blobs);
				try {
					program.build({ device }, options);
					{
						Guard l(x_log);
						loginfo << workerName() << " - " << fname_strm.str() << " sucessfully loaded.\n";
					}
					binaryProgram = program;
					loadedBinary = true;
				} catch (std::exception const&) {
					logerror << workerName() << " - Build info: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
				}
			} else {
				{
					Guard l(x_log);
					logwarn << workerName() << " - Instructed to load binary kernel, but failed to load kernel: " << fname_strm.str() <<
					        endl;
				}
				{
					Guard l(x_log);
					logwarn << workerName() << " - Falling back to OpenCL kernel..." << endl;
				}
			}
		}

		//check whether the current dag fits in memory everytime we recreate the DAG
		cl_ulong result = 0;
		device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
		if (result < dagSize) {
			{
				Guard l(x_log);
				logerror << workerName() << " - OpenCL device " << device.getInfo<CL_DEVICE_NAME>() << " has insufficient GPU memory."
				         << result <<
				         " bytes of memory found < " << dagSize << " bytes of memory required" << endl;
			}
			return false;
		}

		// create buffer for dag
		try {
			{
				Guard l(x_log);
				loginfo << workerName() << " - Creating light cache buffer, size " << light->data().size() << " bytes\n";
			}
			m_light = cl::Buffer(m_context, CL_MEM_READ_ONLY, light->data().size());
			{
				Guard l(x_log);
				loginfo << workerName() << " - Creating DAG buffer, size " << dagSize << " bytes\n";
			}
			m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, dagSize);
			{
				Guard l(x_log);
				loginfo << workerName() << " - Loading kernels" << endl;
			}

			if (s_clKernelName >= CLKernelName::Binary && loadedBinary)
				m_searchKernel = cl::Kernel(binaryProgram, "ethash_search");
			else
				m_searchKernel = cl::Kernel(program, "ethash_search");
			m_dagKernel = cl::Kernel(program, "ethash_calculate_dag_item");
			{
				Guard l(x_log);
				loginfo << workerName() << " - Writing light cache buffer" << endl;
			}
			m_queue.enqueueWriteBuffer(m_light, CL_TRUE, 0, light->data().size(), light->data().data());
		} catch (std::exception const& err) {
			logerror << workerName() << " - Creating DAG buffer failed: " << err.what() << endl;
			return false;
		}
		{
			Guard l(x_log);
			loginfo << workerName() << " - Creating buffer for header." << endl;
		}
		m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(Keccak_RC_kernel));
		m_queue.enqueueWriteBuffer(m_header, CL_TRUE, 0, sizeof(Keccak_RC_kernel), Keccak_RC_kernel);

		m_searchKernel.setArg(1, m_header);
		m_searchKernel.setArg(2, m_dag);

		if (s_clKernelName == CLKernelName::Binary && loadedBinary) {
			const uint32_t epoch = light->light->block_number / ETHASH_EPOCH_LENGTH;
			m_searchKernel.setArg(5, ~0);
			m_searchKernel.setArg(6, dagSize128);
			m_searchKernel.setArg(7, modulo_optimization[epoch].factor);
			m_searchKernel.setArg(8, modulo_optimization[epoch].shift);
			m_searchKernel.setArg(9, m_threadTweak);
		}

		// create mining buffers
		{
			Guard l(x_log);
			loginfo << workerName() << " - Creating mining buffer" << endl;
		}
		m_searchBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, sizeof(search_results));

		uint32_t const work = (uint32_t)(dagSize / sizeof(node));
		uint32_t fullRuns = work / (m_workMultiplier * m_workgroupSize);
		uint32_t const restWork = work % (m_workMultiplier * m_workgroupSize);
		if (restWork > 0) fullRuns++;

		m_dagKernel.setArg(1, m_light);
		m_dagKernel.setArg(2, m_dag);

		auto startDAG = std::chrono::steady_clock::now();
		for (uint32_t i = 0; i < fullRuns; i++) {
			m_dagKernel.setArg(0, i * (m_workMultiplier * m_workgroupSize));
			m_queue.enqueueNDRangeKernel(m_dagKernel, cl::NullRange, m_workMultiplier * m_workgroupSize, m_workgroupSize);
			m_queue.finish();
		}
		auto endDAG = std::chrono::steady_clock::now();

		auto dagTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDAG - startDAG);
		float gb = (float)dagSize / (1024 * 1024 * 1024);
		{
			Guard l(x_log);
			loginfo << workerName() << " - " << gb << " GB of DAG data generated in " << dagTime.count() << " ms." << endl;
		}
	} catch (std::exception const& err) {
		logerror << workerName() << " - OpenCL init failed: " << err.what() << endl;
		throw;
	}
	return true;
}

