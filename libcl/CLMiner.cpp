/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "CLMiner.h"
#include "libethash/internal.h"
#include "libdevcore/Log.h"
#include "CLMiner_kernel.h"
#include <boost/dll.hpp>
#include <boost/multiprecision/cpp_int.hpp>

using namespace dev;
using namespace eth;

using namespace boost::multiprecision;

typedef struct {
	unsigned workGroupSize;
	unsigned workIntensity;
} clConfig;

std::map <std::string, clConfig> optimalConfigs = {
//                      work,  intensity
	{"opencl",          {192,   512}},
	{"ellesmere",       {192,   512}},
	{"baffin",          {192,   512}}
};

namespace dev
{
namespace eth
{

#define MAX_OUTPUTS 0xFFu

typedef  uint32_t search_results[MAX_OUTPUTS + 1];


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

	unsigned Run = 0;

	try {
		while (true) {
			const WorkPackage w = work();
			uint64_t target = 0;

			if (current.header != w.header) {
				// New work received. Update GPU data.
				if (!w) {
					logwarn(workerName() << " - No work. Pause for 3 s.");
					std::this_thread::sleep_for(std::chrono::seconds(3));
					continue;
				}

				if (current.seed != w.seed) {
					if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL) {
						while (s_dagLoadIndex < index)
							this_thread::sleep_for(chrono::seconds(1));
						++s_dagLoadIndex;
					}

					loginfo(workerName() << " - New seed " << w.seed);
					init(w.seed);
					Run = m_workIntensity * m_computeUnits * m_workgroupSize;
				}

				// Upper 64 bits of the boundary.
				target = (uint64_t)(u64)((u256)w.boundary >> 192);

				// Update header constant buffer.
				m_queue.enqueueWriteBuffer(m_header, CL_FALSE, 0, w.header.size, w.header.data());
				m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, MAX_OUTPUTS * sizeof(c_zero), sizeof(c_zero), &c_zero);

				if (w.exSizeBits >= 0) {
					// This can support up to 2^c_log2MaxMiners devices.
					startNonce = w.startNonce | ((uint64_t)index << (64 - LOG2_MAX_MINERS - w.exSizeBits));
				} else
					startNonce = get_start_nonce();

				if (g_logSwitchTime) {
					loginfo(workerName() << " - switch time "
					        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
					                workSwitchStart).count() << " ms.");
				}
				m_searchKernel.setArg(0, m_searchBuffer);  // Supply output buffer to kernel.
				m_searchKernel.setArg(1, m_header);  // Supply header buffer to kernel.
				m_searchKernel.setArg(2, m_dag);  // Supply DAG buffer to kernel.
				m_searchKernel.setArg(3, m_dagSize128);
				m_searchKernel.setArg(5, target);
				m_searchKernel.setArg(6, 0xffffffff);
				m_searchKernel.setArg(7, 1);
			}

			// Read results.
			// TODO: could use pinned host pointer instead.
			uint32_t count, gid;
			m_queue.enqueueReadBuffer(m_searchBuffer, CL_TRUE, MAX_OUTPUTS * sizeof(count), sizeof(count), &count);
			if (count) {
				m_queue.enqueueReadBuffer(m_searchBuffer, CL_TRUE, 0, sizeof(gid), &gid);
				// Reset search buffer if any solution found.
				m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, MAX_OUTPUTS * sizeof(c_zero), sizeof(c_zero), &c_zero);
			}

			// Run the kernel.

			m_searchKernel.setArg(4, startNonce);
			m_queue.enqueueNDRangeKernel(m_searchKernel, cl::NullRange, Run, m_workgroupSize);

			// Report results while the kernel is running.
			if (count) {
				uint64_t nonce = current.startNonce + gid;
				Result r = EthashAux::eval(current.seed, current.header, nonce);
				if (r.value <= current.boundary)
					farm.submitProof(Solution{workerName().c_str(), nonce, r.mixHash, current, current.header != w.header});
				else {
					farm.failedSolution();
					logerror(workerName() << " - discarded incorrect result!");
				}
			}

			current = w;        // kernel now processing newest work
			current.startNonce = startNonce;
			// Increase start nonce for following kernel execution.
			startNonce += Run;

			// Report hash count
			addHashCount(Run);
		}
		m_queue.finish();

	} catch (std::exception const& _e) {
		logerror(workerName() << " - " << _e.what());
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
		logerror("No OpenCL devices found.");
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
			loginfo("Found suitable OpenCL device [" << device.getInfo<CL_DEVICE_NAME>() << "] with " <<
			        result / (1024 * 1024) <<
			        " MB of GPU memory");
			return true;
		}

		logerror("OpenCL device " << device.getInfo<CL_DEVICE_NAME>() << " has insufficient GPU memory." <<
		         result / (1024 * 1024) <<
		         " MB of memory found < " << dagSize << " bytes of memory required");
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
		loginfo(workerName() << " - Platform: " << platformName);

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
			logerror(workerName() << " - No OpenCL devices found.");
			return false;
		}

		// use selected device
		int idx = index % devices.size();
		unsigned deviceId = s_devices[idx] > -1 ? s_devices[idx] : index;
		m_hwmoninfo.deviceIndex = deviceId % devices.size();
		cl::Device& device = devices[deviceId % devices.size()];
		string device_version = device.getInfo<CL_DEVICE_VERSION>();
		string device_name = device.getInfo<CL_DEVICE_NAME>();
		loginfo(workerName() << " - Device: " << device_name << " / " << device_version);
		m_hwmoninfo.deviceName = device.getInfo<CL_DEVICE_VENDOR>() + " " + device_name;
		stringstream ss;
		ss << platformIdx << ':' << m_hwmoninfo.deviceIndex;
		m_hwmoninfo.deviceId = ss.str();

		string clVer = device_version.substr(7, 3);
		if (clVer == "1.0" || clVer == "1.1") {
			if (platformId == OPENCL_PLATFORM_CLOVER) {
				logwarn(workerName() << " - OpenCL " << clVer <<
				        " not supported, but platform Clover might work nevertheless. USE AT OWN RISK!");
			} else {
				logerror(workerName() << " - OpenCL " << clVer << " not supported - minimum required version is 1.2");
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
		m_dagSize128 = (unsigned)(dagSize / ETHASH_MIX_BYTES);

		// patch source code
		// note: The kernels here are simply compiled version of the respective .cl kernels
		// into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
		// See libethash-cl/CMakeLists.txt: add_custom_command()
		// TODO: Just use C++ raw string literal.
		string code;

		if (s_clKernelName == CLKernelName::Opencl) {
			loginfo(workerName() << " - OpenCL kernel: opencl kernel");
			code = string(CLMiner_kernel, CLMiner_kernel + sizeof(CLMiner_kernel));
		} else { // Fallback to experimental kernel if binary loader fails
			loginfo(workerName() << " - OpenCL kernel: " << (s_clKernelName == CLKernelName::Binary ?  "Binary" : "opencl") <<
			        " kernel");
			code = string(CLMiner_kernel, CLMiner_kernel + sizeof(CLMiner_kernel));
		}

		clConfig conf;
		if (s_clKernelName == CLKernelName::Opencl)
			conf = optimalConfigs["opencl"];
		else { /* if (s_clKernelName == CLKernelName::Binary) */
			std::string name = device.getInfo<CL_DEVICE_NAME>();
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);
			if (optimalConfigs.find(name) == optimalConfigs.end()) {
				logerror(workerName() << " - Can't find configuration for binary kernel " << name);
				throw runtime_error("No kernel");
			}
			conf = optimalConfigs[name];
		}
		m_workgroupSize = conf.workGroupSize;
		m_workIntensity = conf.workIntensity;

		m_computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		// Apparently some 36 CU devices return a bogus 14!!!
		m_computeUnits = m_computeUnits == 14 ? 36 : m_computeUnits;
		logwarn(workerName()
		        << " - work group " << m_workgroupSize
		        << ", work intensity " << m_workIntensity);

		uint32_t lightSize64 = (unsigned)(light->data().size() / sizeof(node));
		addDefinition(code, "WORKSIZE", m_workgroupSize);
		//addDefinition(code, "DAG_SIZE", m_dagSize128);
		//addDefinition(code, "LIGHT_SIZE", lightSize64);
		addDefinition(code, "PLATFORM", platformId);
		addDefinition(code, "COMPUTE", computeCapability);

		// create miner OpenCL program
		cl::Program::Sources sources{{code.data(), code.size()}};
		cl::Program program(m_context, sources), binaryProgram;
		try {
			program.build({device}, options);
		} catch (std::exception const&) {
			logerror(workerName() << " - Build info: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
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
			fname_strm << boost::dll::program_location().parent_path().string() << "/kernels/" << name << m_workgroupSize << ".bin";

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
					loginfo(workerName() << " - " << fname_strm.str() << " sucessfully loaded.");
					binaryProgram = program;
					loadedBinary = true;
				} catch (std::exception const&) {
					logerror(workerName() << " - Build info: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
				}
			} else {
				logwarn(workerName() << " - Instructed to load binary kernel, but failed to load kernel: " << fname_strm.str());
				logwarn(workerName() << " - Falling back to OpenCL kernel...");
			}
		}

		//check whether the current dag fits in memory everytime we recreate the DAG
		cl_ulong result = 0;
		device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
		if (result < dagSize) {
			logerror(workerName() << " - OpenCL device " << device.getInfo<CL_DEVICE_NAME>() << " has insufficient GPU memory."
			         << result / (1024 * 1024) <<
			         " MB of memory found < " << dagSize << " bytes of memory required");
			return false;
		}

		// create buffer for dag
		try {
			loginfo(workerName() << " - Creating light cache buffer, size " << light->data().size() / 1024 << " KB");
			m_light = cl::Buffer(m_context, CL_MEM_READ_ONLY, light->data().size());
			loginfo(workerName() << " - Creating DAG buffer, size " << dagSize / (1024 * 1024) << " MB");
			m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, dagSize);
			loginfo(workerName() << " - Loading kernels");

			if (s_clKernelName >= CLKernelName::Binary && loadedBinary) {
				m_searchKernel = cl::Kernel(binaryProgram, "search");
				m_dagKernel = cl::Kernel(binaryProgram, "GenerateDAG");
			} else {
				m_searchKernel = cl::Kernel(program, "search");
				m_dagKernel = cl::Kernel(program, "GenerateDAG");
			}
			loginfo(workerName() << " - Writing light cache buffer");
			m_queue.enqueueWriteBuffer(m_light, CL_TRUE, 0, light->data().size(), light->data().data());
		} catch (std::exception const& err) {
			logerror(workerName() << " - Creating DAG buffer failed: " << err.what());
			return false;
		}
		loginfo(workerName() << " - Creating buffer for header.");
		m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, headerBuffer.size);
		m_queue.enqueueWriteBuffer(m_header, CL_TRUE, 0, headerBuffer.size, headerBuffer.data());

		// create mining buffers
		loginfo(workerName() << " - Creating mining buffer");
		m_searchBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, sizeof(search_results));

		uint32_t const work = (uint32_t)(dagSize / sizeof(node));
		uint32_t Run = m_workIntensity * m_computeUnits * m_workgroupSize;

		m_dagKernel.setArg(1, m_light);
		m_dagKernel.setArg(2, m_dag);
		m_dagKernel.setArg(3, lightSize64);
		m_dagKernel.setArg(4, m_dagSize128 * 2);
		m_dagKernel.setArg(5, 0xffffffff);
		auto startDAG = std::chrono::steady_clock::now();
		for (uint32_t i = 0; i < work; i += Run) {
			m_dagKernel.setArg(0, i);
			m_queue.enqueueNDRangeKernel(m_dagKernel, cl::NullRange, Run, m_workgroupSize);
			m_queue.finish();
		}
		auto endDAG = std::chrono::steady_clock::now();

		if (0) {
			uint8_t* buf = new uint8_t[dagSize];
			m_queue.enqueueReadBuffer(m_dag, CL_TRUE, 0, dagSize, buf);
			std::fstream myfile;
			myfile = std::fstream("dag2.binary", std::ios::out | std::ios::binary);
			myfile.write((char*)buf, dagSize);
			myfile.close();
			delete [] buf;
		}

		auto dagTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDAG - startDAG);
		float gb = (float)dagSize / (1024 * 1024 * 1024);
		loginfo(workerName() << " - " << gb << " GB of DAG data generated in " << dagTime.count() << " ms.");
	} catch (std::exception const& err) {
		logerror(workerName() << " - OpenCL init failed: " << err.what());
		throw;
	}
	return true;
}

