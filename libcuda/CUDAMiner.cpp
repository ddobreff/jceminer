/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "CUDAMiner.h"
#include "libdevcore/Log.h"

using namespace std;
using namespace dev;
using namespace eth;

unsigned CUDAMiner::s_numInstances = 0;

vector<int> CUDAMiner::s_devices(MAX_MINERS, -1);

CUDAMiner::CUDAMiner(FarmFace& _farm, unsigned _index) :
	Miner("cuda-", _farm, _index),
	m_light(getNumDevices()) {}

CUDAMiner::~CUDAMiner()
{
}

bool CUDAMiner::init(const h256& seed)
{
	try {
		if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
			while (s_dagLoadIndex < index)
				this_thread::sleep_for(chrono::milliseconds(100));
		unsigned device = s_devices[index] > -1 ? s_devices[index] : index;

		{
			Guard l(x_log);
			loginfo << workerName() << " - Initialising miner " << index << endl;
		}

		EthashAux::LightType light;
		light = EthashAux::light(seed);
		bytesConstRef lightData = light->data();

		cuda_init(getNumDevices(), light->light, lightData.data(), lightData.size(),
		          device, (s_dagLoadMode == DAG_LOAD_MODE_SINGLE), s_dagInHostMemory, s_dagCreateDevice);
		s_dagLoadIndex++;

		if (s_dagLoadMode == DAG_LOAD_MODE_SINGLE) {
			if (s_dagLoadIndex >= s_numInstances && s_dagInHostMemory) {
				// all devices have loaded DAG, we can free now
				delete[] s_dagInHostMemory;
				s_dagInHostMemory = NULL;
				{
					Guard l(x_log);
					loginfo << workerName() << " - Freeing DAG from host" << endl;
				}
			}
		}
		return true;
	} catch (std::exception const& _e) {
		logerror << workerName() << " - Error CUDA mining: " << _e.what() << endl;
		throw;
	}
}

void CUDAMiner::workLoop()
{
	WorkPackage current;
	current.header = h256{1u};
	current.seed = h256{1u};

	try {
		while (true) {
			// take local copy of work since it may end up being overwritten.
			const WorkPackage w = work();

			if (current.header != w.header || current.seed != w.seed) {
				if (!w || w.header == h256()) {
					{
						Guard l(x_log);
						logwarn << workerName() << " - No work. Pause for 3 s." << endl;
					}
					std::this_thread::sleep_for(std::chrono::seconds(3));
					continue;
				}
				if (current.seed != w.seed)
					if (!init(w.seed))
						break;
				current = w;
			}
			uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);
			uint64_t startN = current.startNonce;
			if (current.exSizeBits >= 0) {
				// this can support up to 2^MAX_GPU devices
				startN = current.startNonce | ((uint64_t)index << (64 - LOG2_MAX_MINERS - current.exSizeBits));
			}
			search(current.header.data(), upper64OfBoundary, (current.exSizeBits >= 0), startN, w);
		}

		// Reset miner and stop working
		CUDA_SAFE_CALL(cudaDeviceReset());
	} catch (std::exception const& _e) {
		logerror << workerName() << " - Fatal GPU error: " << _e.what() << endl;
		throw;
	}
}

void CUDAMiner::kick_miner()
{
	m_new_work.store(true, memory_order_relaxed);
}

void CUDAMiner::setNumInstances(unsigned _instances)
{
	s_numInstances = std::min<unsigned>(_instances, getNumDevices());
}

void CUDAMiner::setDevices(const vector<unsigned>& _devices, unsigned _selectedDeviceCount)
{
	for (unsigned i = 0; i < _selectedDeviceCount; i++)
		s_devices[i] = _devices[i];
}

unsigned CUDAMiner::getNumDevices()
{
	int deviceCount = -1;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err == cudaSuccess)
		return deviceCount;

	if (err == cudaErrorInsufficientDriver) {
		int driverVersion = -1;
		cudaDriverGetVersion(&driverVersion);
		if (driverVersion == 0)
			throw std::runtime_error{"No CUDA driver found"};
		throw std::runtime_error{"Insufficient CUDA driver: " + std::to_string(driverVersion)};
	}

	throw std::runtime_error{cudaGetErrorString(err)};
}

void CUDAMiner::listDevices()
{
	cout << "\nListing CUDA devices.\nFORMAT: [deviceID] deviceName\n";
	try {
		int numDevices = getNumDevices();
		for (int i = 0; i < numDevices; ++i) {
			cudaDeviceProp props;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));

			cout << "[" + to_string(i) + "] " + string(props.name) + "\n";
			cout << "\tCompute version: " + to_string(props.major) + "." + to_string(props.minor) + "\n";
			cout << "\tcudaDeviceProp::totalGlobalMem: " + to_string(props.totalGlobalMem) + "\n";
			cout << "\tPci: " << setw(4) << setfill('0') << hex << props.pciDomainID << ':' << setw(2)
			     << props.pciBusID << ':' << setw(2) << props.pciDeviceID << '\n';
		}
	} catch (std::exception const&) {
	}
}

bool CUDAMiner::configureGPU(
    unsigned _blockSize,
    unsigned _gridSize,
    unsigned _numStreams,
    unsigned _scheduleFlag,
    uint64_t _currentBlock,
    unsigned _dagLoadMode,
    unsigned _dagCreateDevice,
    bool _eval
)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;

	if (!cuda_configureGPU(
	        getNumDevices(),
	        s_devices,
	        ((_blockSize + 7) / 8) * 8,
	        _gridSize,
	        _numStreams,
	        _scheduleFlag,
	        _currentBlock,
	        _eval)
	   ) {
		cout << "No CUDA device with sufficient memory was found. Can't CUDA mine. Remove the -U argument" << endl;
		return false;
	}
	return true;
}

void CUDAMiner::setParallelHash(unsigned _parallelHash)
{
	m_parallelHash = _parallelHash;
}

bool CUDAMiner::cuda_configureGPU(
    size_t numDevices,
    const vector<int>& _devices,
    unsigned _blockSize,
    unsigned _gridSize,
    unsigned _numStreams,
    unsigned _scheduleFlag,
    uint64_t _currentBlock,
    bool _eval
)
{
	try {
		s_blockSize = _blockSize;
		s_gridSize = _gridSize;
		s_numStreams = _numStreams;
		s_scheduleFlag = _scheduleFlag;
		s_eval = _eval;

		{
			Guard l(x_log);
			loginfo << "Using grid size " << s_gridSize << ", block size " << s_blockSize << endl;
		}

		// by default let's only consider the DAG of the first epoch
		uint64_t dagSize = ethash_get_datasize(_currentBlock);
		int devicesCount = static_cast<int>(numDevices);
		for (int i = 0; i < devicesCount; i++) {
			if (_devices[i] != -1) {
				int deviceId = min(devicesCount - 1, _devices[i]);
				cudaDeviceProp props;
				CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, deviceId));
				if (props.totalGlobalMem >= dagSize) {
					Guard l(x_log);
					loginfo <<  "Found suitable CUDA device [" << string(props.name) << "] with " << props.totalGlobalMem <<
					        " bytes of GPU memory" << endl;
				} else {
					{
						Guard l(x_log);
						logerror << "CUDA device " << string(props.name) << " has insufficient GPU memory." << props.totalGlobalMem <<
						         " bytes of memory found < " << dagSize << " bytes of memory required" << endl;
					}
					return false;
				}
			}
		}
		return true;
	} catch (std::exception const& _e) {
		logerror << "Error CUDA mining: " << _e.what() << endl;
		throw;
	}
}

unsigned CUDAMiner::m_parallelHash;
unsigned CUDAMiner::s_blockSize;
unsigned CUDAMiner::s_gridSize;
unsigned CUDAMiner::s_numStreams;
unsigned CUDAMiner::s_scheduleFlag;
bool CUDAMiner::s_eval = false;

bool CUDAMiner::cuda_init(
    size_t numDevices,
    ethash_light_t _light,
    uint8_t const* _lightData,
    uint64_t _lightSize,
    unsigned _deviceId,
    bool _cpyToHost,
    uint8_t*& hostDAG,
    unsigned dagCreateDevice)
{
	try {
		if (numDevices == 0)
			return false;

		// use selected device
		m_device_num = _deviceId < numDevices - 1 ? _deviceId : numDevices - 1;
		m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
		m_hwmoninfo.indexSource = HwMonitorIndexSource::CUDA;
		m_hwmoninfo.deviceIndex = m_device_num;

		cudaDeviceProp device_props;
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, m_device_num));

		{
			Guard l(x_log);
			loginfo << workerName() << " - Using device: " << device_props.name << " (Compute " + to_string(
			            device_props.major) + "." + to_string(
			            device_props.minor) + ")" << endl;
		}

		m_search_buf = new volatile search_results *[s_numStreams];
		m_streams = new cudaStream_t[s_numStreams];

		uint64_t dagSize = ethash_get_datasize(_light->block_number);
		uint32_t dagSize128   = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(_lightSize / sizeof(node));



		CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
		{
			Guard l(x_log);
			loginfo << workerName() << " - Set Device to current" << endl;
		}
		if (dagSize128 != m_dag_size || !m_dag) {
			//Check whether the current device has sufficient memory everytime we recreate the dag
			if (device_props.totalGlobalMem < dagSize) {
				{
					Guard l(x_log);
					logerror <<  workerName() << " - CUDA device " << string(device_props.name) << " has insufficient GPU memory." <<
					         device_props.totalGlobalMem << " bytes of memory found < " << dagSize << " bytes of memory required" << endl;
				}
				return false;
			}
			//We need to reset the device and recreate the dag
			{
				Guard l(x_log);
				logwarn << workerName() << " - Resetting device" << endl;
			}
			CUDA_SAFE_CALL(cudaDeviceReset());
			{
				Guard l(x_log);
				loginfo << workerName() << " - Device successfully reset" << endl;
			}
			CUDA_SAFE_CALL(cudaSetDeviceFlags(s_scheduleFlag));
			CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
			//We need to reset the light and the Dag for the following code to reallocate
			//since cudaDeviceReset() free's all previous allocated memory
			m_light[m_device_num] = nullptr;
			m_dag = nullptr;
		}
		// create buffer for cache
		hash128_t* dag = m_dag;
		hash64_t* light = m_light[m_device_num];

		if (!light) {
			{
				Guard l(x_log);
				loginfo << workerName() << " - Allocating light with size: " << _lightSize << " bytes\n";
			}
			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), _lightSize));
		}
		// copy lightData to device
		CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), _lightData, _lightSize, cudaMemcpyHostToDevice));
		m_light[m_device_num] = light;

		if (dagSize128 != m_dag_size || !dag) // create buffer for dag
			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagSize));

		set_constants(dag, dagSize128, light, lightSize64); //in ethash_cuda_miner_kernel.cu

		if (dagSize128 != m_dag_size || !dag) {
			// create mining buffers
			for (unsigned i = 0; i != s_numStreams; ++i) {
				CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], sizeof(search_results)));
				CUDA_SAFE_CALL(cudaStreamCreate(&m_streams[i]));
			}

			memset(&m_current_header, 0, sizeof(hash32_t));
			m_current_target = 0;
			m_current_nonce = 0;
			m_current_index = 0;

			if (!hostDAG) {
				if ((m_device_num == dagCreateDevice) || !_cpyToHost) { //if !cpyToHost -> All devices shall generate their DAG
					{
						Guard l(x_log);
						loginfo << workerName() << " - Generating DAG for GPU" << m_device_num << " with dagSize: " << dagSize << " bytes\n";
					}
					ethash_generate_dag(dagSize, s_gridSize, s_blockSize, m_streams[0]);

					if (_cpyToHost) {
						uint8_t* memoryDAG = new uint8_t[dagSize];
						{
							Guard l(x_log);
							loginfo << workerName() << " - Copying DAG from GPU" << m_device_num << " to host" << endl;
						}
						CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(memoryDAG), dag, dagSize, cudaMemcpyDeviceToHost));

						hostDAG = memoryDAG;
					}
				} else {
					while (!hostDAG)
						this_thread::sleep_for(chrono::milliseconds(100));
					goto cpyDag;
				}
			} else {
			cpyDag: {
					Guard l(x_log);
					loginfo << workerName() << " - Copying DAG from host to GPU" << m_device_num << endl;
				}
				const void* hdag = (const void*)hostDAG;
				CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(dag), hdag, dagSize, cudaMemcpyHostToDevice));
			}
		}

		m_dag = dag;
		m_dag_size = dagSize128;
		return true;
	} catch (std::exception const& _e) {
		logerror << workerName() << " - Error CUDA mining: " << _e.what() << endl;
		throw;
	}
}

void CUDAMiner::search(
    uint8_t const* header,
    uint64_t target,
    bool _ethStratum,
    uint64_t _startN,
    const dev::eth::WorkPackage& w)
{
	bool initialize = false;
	if ((memcmp(&m_current_header, header, sizeof(hash32_t))) || (m_current_target != target)) {
		m_current_header = *reinterpret_cast<hash32_t const*>(header);
		m_current_target = target;
		set_header_and_target(m_current_header, m_current_target);
		initialize = true;
	}
	if (_ethStratum) {
		if (initialize) {
			m_starting_nonce = 0;
			m_current_index = 0;
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			for (unsigned int i = 0; i < s_numStreams; i++)
				m_search_buf[i]->count = 0;
		}
		if (m_starting_nonce != _startN) {
			// reset nonce counter
			m_starting_nonce = _startN;
			m_current_nonce = m_starting_nonce;
		}
	} else {
		if (initialize) {
			m_current_nonce = get_start_nonce();
			m_current_index = 0;
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			for (unsigned int i = 0; i < s_numStreams; i++)
				m_search_buf[i]->count = 0;
		}
	}
	const uint32_t batch_size = s_gridSize * s_blockSize;
	volatile search_results* buffer;
	uint32_t stream_index;
	while (m_current_index < s_numStreams) {
		stream_index = m_current_index % s_numStreams;
		cudaStream_t stream = m_streams[stream_index];
		buffer = m_search_buf[stream_index];
		run_ethash_search(s_gridSize, s_blockSize, stream, buffer, m_current_nonce, m_parallelHash);
		m_current_index++;
	}

	while (true) {
		m_current_nonce += batch_size;
		stream_index = m_current_index % s_numStreams;
		cudaStream_t stream = m_streams[stream_index];
		buffer = m_search_buf[stream_index];
		uint32_t found_count = 0;
		uint64_t nonce;
		h256 mix;
		uint64_t nonce_base = m_current_nonce - s_numStreams * batch_size;
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
		found_count = buffer->count;
		if (found_count) {
			nonce = nonce_base + buffer->gid;
			if (!s_eval)
				memcpy(mix.data(), (const void*)buffer->mix, sizeof(buffer->mix));
			buffer->count = 0;
		}
		run_ethash_search(s_gridSize, s_blockSize, stream, buffer, m_current_nonce, m_parallelHash);
		if (found_count) {
			if (!s_eval) {
				farm.submitProof(workerName(), Solution{nonce, mix, w, m_new_work});
			} else {
				Result r = EthashAux::eval(w.seed, w.header, nonce);
				if (r.value < w.boundary)
					farm.submitProof(workerName(), Solution{nonce, r.mixHash, w, m_new_work});
				else {
					farm.failedSolution();
					{
						Guard l(x_log);
						logwarn << workerName() << " - Incorrect result discarded!\n";
					}
				}
			}
		}

		addHashCount(batch_size);
		bool t = true;
		if (m_new_work.compare_exchange_strong(t, false, memory_order_relaxed)) {
			if (g_logSwitchTime) {
				Guard l(x_log);
				loginfo << workerName() << " - switch time " << std::chrono::duration_cast<std::chrono::milliseconds>
				        (std::chrono::high_resolution_clock::now() - workSwitchStart).count() << " ms." << endl;
			}
			break;
		}
		m_current_index++;
	}
}

