/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <fstream>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"

// macOS OpenCL fix:
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#endif

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA  1
#define OPENCL_PLATFORM_AMD     2
#define OPENCL_PLATFORM_CLOVER  3


namespace dev
{
namespace eth
{

enum CLKernelName : unsigned int {
	Opencl = 0,
	Binary,
};

class CLMiner: public Miner
{
public:

	CLMiner(FarmFace& _farm, unsigned _index);
	~CLMiner() override;

	static unsigned instances()
	{
		return s_numInstances > 0 ? s_numInstances : 1;
	}
	static unsigned getNumDevices();
	static void listDevices();
	static bool configureGPU(
	    unsigned _platformId,
	    unsigned _dagLoadMode,
	    unsigned _dagCreateDevice,
	    bool _eval
	);
	static void setNumInstances(unsigned _instances)
	{
		s_numInstances = std::min<unsigned>(_instances, getNumDevices());
	}
	static void setDevices(const vector<unsigned>& _devices, unsigned _selectedDeviceCount)
	{
		for (unsigned i = 0; i < _selectedDeviceCount; i++)
			s_devices[i] = _devices[i];
	}
	static void setCLKernel(unsigned _clKernel)
	{
		s_clKernelName = (CLKernelName)_clKernel;
	}
protected:
	void kick_miner() override;

private:
	void workLoop() override;

	bool init(const h256& seed);

	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Kernel m_searchKernel;
	cl::Kernel m_dagKernel;
	cl::Buffer m_dag;
	cl::Buffer m_light;
	cl::Buffer m_header;
	cl::Buffer m_searchBuffer;
	unsigned m_dagSize128;
	unsigned m_workIntensity;
	unsigned m_workgroupSize;
	unsigned m_computeUnits;

	static bool s_eval;
	static unsigned s_platformId;
	static unsigned s_numInstances;
	static CLKernelName s_clKernelName;
	static vector<int> s_devices;
};

static const h256  headerBuffer;

}
}
