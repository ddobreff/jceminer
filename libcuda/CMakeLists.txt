find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo)

list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
#list(APPEND CUDA_NVCC_FLAGS_DEBUG -G)
list(APPEND CUDA_NVCC_FLAGS_DEBUG -O3)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	"-gencode arch=compute_60,code=sm_60"
	"-gencode arch=compute_61,code=sm_61"
	"-gencode arch=compute_62,code=sm_62"
)

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.h" "*.cuh")

cuda_add_library(cuda STATIC ${sources} ${headers})
target_link_libraries(cuda ethcore ethash)
target_include_directories(cuda PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
