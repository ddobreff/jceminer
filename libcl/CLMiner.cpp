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

#include "CLMiner.h"
#include "libethash/internal.h"
#include "libdevcore/Log.h"
#include "CLMiner_kernel_stable.h"
#include "CLMiner_kernel_experimental.h"
#include <boost/dll.hpp>

using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{

unsigned CLMiner::s_workgroupSize;
unsigned CLMiner::s_initialGlobalWorkSize;
unsigned CLMiner::s_threadsPerHash;
unsigned CLMiner::s_threadTweak;
CLKernelName CLMiner::s_clKernelName;

constexpr size_t c_maxSearchResults = 1;

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

unsigned CLMiner::s_platformId = 0;
unsigned CLMiner::s_numInstances = 0;
vector<int> CLMiner::s_devices(MAX_MINERS, -1);

CLMiner::CLMiner(FarmFace& _farm, unsigned _index):
	Miner("cl-", _farm, _index)
{}

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
						logwarn << "No work. Pause for 3 s." << endl << flush;
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
						loginfo << "New seed " << w.seed << endl << flush;
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

				// FIXME: This logic should be move out of here.
				if (w.exSizeBits >= 0) {
					// This can support up to 2^c_log2MaxMiners devices.
					startNonce = w.startNonce | ((uint64_t)index << (64 - LOG2_MAX_MINERS - w.exSizeBits));
				} else
					startNonce = get_start_nonce();

				if (g_logSwitchTime) {
					Guard l(x_log);
					loginfo << workerName() << " - switch time "
					        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
					                workSwitchStart).count() << " ms." << endl << flush;
				}
			}

			// Read results.
			// TODO: could use pinned host pointer instead.
			uint32_t results[c_maxSearchResults + 1];
			m_queue.enqueueReadBuffer(m_searchBuffer, CL_TRUE, 0, sizeof(results), &results);

			uint64_t nonce = 0;
			if (results[0] > 0) {
				// Ignore results except the first one.
				nonce = current.startNonce + results[1];
				// Reset search buffer if any solution found.
				m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(c_zero), &c_zero);
			}

			// Run the kernel.
			m_searchKernel.setArg(3, startNonce);
			m_queue.enqueueNDRangeKernel(m_searchKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);

			// Report results while the kernel is running.
			// It takes some time because ethash must be re-evaluated on CPU.
			if (nonce != 0) {
				Result r = EthashAux::eval(current.seed, current.header, nonce);
				if (r.value < current.boundary)
					farm.submitProof(workerName(), Solution{nonce, r.mixHash, current, current.header != w.header});
				else {
					farm.failedSolution();
					{
						Guard l(x_log);
						logwarn << "FAILURE: GPU gave incorrect result! Discarded." << endl << flush;
					}
				}
			}

			current = w;        // kernel now processing newest work
			current.startNonce = startNonce;
			// Increase start nonce for following kernel execution.
			startNonce += m_globalWorkSize;

			// Report hash count
			addHashCount(m_globalWorkSize);
		}

		// Make sure the last buffer write has finished --
		// it reads local variable.
		m_queue.finish();
	} catch (std::exception const& _e) {
		logerror << _e.what() << endl << flush;
		exit(-1);
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
			logerror << "No OpenCL devices found." << endl << flush;
		}
		return 0;
	}
	return devices.size();
}

void CLMiner::listDevices()
{
	string outString = "\nListing OpenCL devices.\nFORMAT: [platformID] [deviceID] device.getInfo<CL_DEVICE_VERSION>()\n";
	unsigned int i = 0;

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
}

bool CLMiner::configureGPU(
    unsigned _localWorkSize,
    unsigned _globalWorkSizeMultiplier,
    unsigned _platformId,
    uint64_t _currentBlock,
    unsigned _dagLoadMode,
    unsigned _dagCreateDevice
)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;

	s_platformId = _platformId;

	_localWorkSize = ((_localWorkSize + 7) / 8) * 8;
	s_workgroupSize = _localWorkSize;

	s_initialGlobalWorkSize = _globalWorkSizeMultiplier * _localWorkSize;

	uint64_t dagSize = ethash_get_datasize(_currentBlock);

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
				        " bytes of GPU memory" << endl << flush;
			}
			return true;
		}

		{
			Guard l(x_log);
			logerror << "OpenCL device " << device.getInfo<CL_DEVICE_NAME>() << " has insufficient GPU memory." << result <<
			         " bytes of memory found < " << dagSize << " bytes of memory required" << endl << flush;
		}
	}

	cout << "No GPU device with sufficient memory was found. Can't GPU mine. Remove the -G argument" << endl << flush;
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
			loginfo << "Platform: " << platformName << endl << flush;
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
				logerror << "No OpenCL devices found." << endl << flush;
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
			loginfo << "Device: " << device.getInfo<CL_DEVICE_NAME>() << " / " << device_version << endl << flush;
		}

		string clVer = device_version.substr(7, 3);
		if (clVer == "1.0" || clVer == "1.1") {
			if (platformId == OPENCL_PLATFORM_CLOVER) {
				Guard l(x_log);
				logwarn << "OpenCL " << clVer << " not supported, but platform Clover might work nevertheless. USE AT OWN RISK!" <<
				        endl << flush;
			} else {
				{
					Guard l(x_log);
					logerror << "OpenCL " << clVer << " not supported - minimum required version is 1.2" << endl << flush;
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

		m_workgroupSize = s_workgroupSize;
		m_globalWorkSize = s_initialGlobalWorkSize;

		// Adjust global work size according to number of CUs
		cl_uint maxCUs;
		clGetDeviceInfo(device(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxCUs, nullptr);
		{
			Guard l(x_log);
			logwarn << "Global work size adjusted for " << maxCUs << " CUs" << endl << flush;
		}
		{
			Guard l(x_log);
			logwarn << "Specified Global work size: " << m_globalWorkSize << endl << flush;
		}
		m_globalWorkSize = (m_globalWorkSize * maxCUs) / 36;
		{
			Guard l(x_log);
			logwarn << "Adjusted global work size: " << m_globalWorkSize << endl << flush;
		}
		if (m_globalWorkSize % m_workgroupSize != 0)
			m_globalWorkSize = ((m_globalWorkSize / m_workgroupSize) + 1) * m_workgroupSize;

		uint64_t dagSize = ethash_get_datasize(light->light->block_number);
		uint32_t dagSize128 = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(light->data().size() / sizeof(node));

		// patch source code
		// note: The kernels here are simply compiled version of the respective .cl kernels
		// into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
		// See libethash-cl/CMakeLists.txt: add_custom_command()
		// TODO: Just use C++ raw string literal.
		string code;

		if (s_clKernelName == CLKernelName::Experimental) {
			{
				Guard l(x_log);
				loginfo << "OpenCL kernel: Experimental kernel" << endl << flush;
			}
			code = string(CLMiner_kernel_experimental, CLMiner_kernel_experimental + sizeof(CLMiner_kernel_experimental));
		} else { // Fallback to experimental kernel if binary loader fails
			{
				Guard l(x_log);
				loginfo << "OpenCL kernel: " << (s_clKernelName == CLKernelName::Binary ?  "Binary" : "Experimental") << " kernel" <<
				        endl << flush;
			}

			//CLMiner_kernel_stable.cl will do a #undef THREADS_PER_HASH
			if (s_threadsPerHash != 8) {
				//
				{
					Guard l(x_log);
					logwarn << "The current stable OpenCL kernel only supports exactly 8 threads. Thread parameter will be ignored." <<
					        endl << flush;
				}
			}

			code = string(CLMiner_kernel_stable, CLMiner_kernel_stable + sizeof(CLMiner_kernel_stable));
		}
		addDefinition(code, "GROUP_SIZE", m_workgroupSize);
		addDefinition(code, "DAG_SIZE", dagSize128);
		addDefinition(code, "LIGHT_SIZE", lightSize64);
		addDefinition(code, "ACCESSES", ETHASH_ACCESSES);
		addDefinition(code, "MAX_OUTPUTS", c_maxSearchResults);
		addDefinition(code, "PLATFORM", platformId);
		addDefinition(code, "COMPUTE", computeCapability);
		addDefinition(code, "THREADS_PER_HASH", s_threadsPerHash);

		// create miner OpenCL program
		cl::Program::Sources sources{{code.data(), code.size()}};
		cl::Program program(m_context, sources), binaryProgram;
		try {
			program.build({device}, options);
		} catch (std::exception const&) {
			logerror << "Build info: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl << flush;
			return false;
		}

		// If we have a binary kernel, we load it in tandem with the opencl,
		// that way, we can use the dag generate opencl code
		bool loadedBinary = false;
		unsigned int computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

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
						loginfo << fname_strm.str() << " sucessfully loaded.\n" << flush;
					}
					binaryProgram = program;
					loadedBinary = true;
				} catch (std::exception const&) {
					logerror << "Build info: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl << flush;
				}

				computeUnits = computeUnits == 14 ? 36 : computeUnits;
				m_globalWorkSize = computeUnits << 17;
			} else {
				{
					Guard l(x_log);
					logwarn << "Instructed to load binary kernel, but failed to load kernel: " << fname_strm.str() << endl << flush;
				}
				{
					Guard l(x_log);
					logwarn << "Falling back to OpenCL kernel..." << endl << flush;
				}
			}
		}

		//check whether the current dag fits in memory everytime we recreate the DAG
		cl_ulong result = 0;
		device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
		if (result < dagSize) {
			{
				Guard l(x_log);
				logerror << "OpenCL device " << device.getInfo<CL_DEVICE_NAME>() << " has insufficient GPU memory." << result <<
				         " bytes of memory found < " << dagSize << " bytes of memory required" << endl << flush;
			}
			return false;
		}

		// create buffer for dag
		try {
			{
				Guard l(x_log);
				loginfo << "Creating light cache buffer, size " << light->data().size() << " bytes\n" << flush;
			}
			m_light = cl::Buffer(m_context, CL_MEM_READ_ONLY, light->data().size());
			{
				Guard l(x_log);
				loginfo << "Creating DAG buffer, size " << dagSize << " bytes\n" << flush;
			}
			m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, dagSize);
			{
				Guard l(x_log);
				loginfo << "Loading kernels" << endl << flush;
			}

			if (s_clKernelName >= CLKernelName::Binary && loadedBinary)
				m_searchKernel = cl::Kernel(binaryProgram, "ethash_search");
			else
				m_searchKernel = cl::Kernel(program, "ethash_search");
			m_dagKernel = cl::Kernel(program, "ethash_calculate_dag_item");
			{
				Guard l(x_log);
				loginfo << "Writing light cache buffer" << endl << flush;
			}
			m_queue.enqueueWriteBuffer(m_light, CL_TRUE, 0, light->data().size(), light->data().data());
		} catch (std::exception const& err) {
			logerror << "Creating DAG buffer failed: " << err.what() << endl << flush;
			return false;
		}
		{
			Guard l(x_log);
			loginfo << "Creating buffer for header." << endl << flush;
		}
		m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(Keccak_RC_kernel));
		m_queue.enqueueWriteBuffer(m_header, CL_TRUE, 0, sizeof(Keccak_RC_kernel), Keccak_RC_kernel);


		m_searchKernel.setArg(1, m_header);
		m_searchKernel.setArg(2, m_dag);
		m_searchKernel.setArg(5, ~0u);  // Pass this to stop the compiler unrolling the loops.

		if (s_clKernelName >= CLKernelName::Binary && loadedBinary) {
			const uint32_t epoch = light->light->block_number / ETHASH_EPOCH_LENGTH;
			m_searchKernel.setArg(6, dagSize128);
			m_searchKernel.setArg(7, modulo_optimization[epoch].factor);
			m_searchKernel.setArg(8, modulo_optimization[epoch].shift);
			m_searchKernel.setArg(9, s_threadTweak);
		}

		// create mining buffers
		{
			Guard l(x_log);
			loginfo << "Creating mining buffer" << endl << flush;
		}
		m_searchBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, (c_maxSearchResults + 1) * sizeof(uint32_t));

		uint32_t const work = (uint32_t)(dagSize / sizeof(node));
		uint32_t fullRuns = work / m_globalWorkSize;
		uint32_t const restWork = work % m_globalWorkSize;
		if (restWork > 0) fullRuns++;

		m_dagKernel.setArg(1, m_light);
		m_dagKernel.setArg(2, m_dag);
		m_dagKernel.setArg(3, ~0u);

		auto startDAG = std::chrono::steady_clock::now();
		for (uint32_t i = 0; i < fullRuns; i++) {
			m_dagKernel.setArg(0, i * m_globalWorkSize);
			m_queue.enqueueNDRangeKernel(m_dagKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);
			m_queue.finish();
		}
		auto endDAG = std::chrono::steady_clock::now();

		auto dagTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDAG - startDAG);
		float gb = (float)dagSize / (1024 * 1024 * 1024);
		{
			Guard l(x_log);
			loginfo << gb << " GB of DAG data generated in " << dagTime.count() << " ms." << endl << flush;
		}
	} catch (std::exception const& err) {
		logerror << "OpenCL init failed: " << err.what() << endl << flush;
		exit(-1);
	}
	return true;
}

