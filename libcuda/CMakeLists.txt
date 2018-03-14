find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo)

list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
list(APPEND CUDA_NVCC_FLAGS_DEBUG -G)

if(COMPUTE AND (COMPUTE GREATER 0))
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE}")
else()
	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}
		"-gencode arch=compute_60,code=sm_60"
		"-gencode arch=compute_61,code=sm_61"
		"-gencode arch=compute_62,code=sm_62"
		"-gencode arch=compute_70,code=sm_70"
	)
endif()

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.h" "*.cuh")

cuda_add_library(cuda STATIC ${sources} ${headers})
target_link_libraries(cuda ethcore ethash)
target_include_directories(cuda PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
