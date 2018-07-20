cmake_minimum_required(VERSION 3.3)

include(cmake/cable/bootstrap.cmake)

include(CableBuildInfo)
include(CableBuildType)
include(CableToolchains)

cable_configure_toolchain(DEFAULT cxx11)

set(HUNTER_CONFIGURATION_TYPES Release)
set(HUNTER_CACHE_SERVERS "https://github.com/ethereum/hunter-cache")
include(HunterGate)
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.19.213.tar.gz"
    SHA1 "59b2b880178bdc1e9cce83f279f1d88feefa07a2"
    LOCAL
)

if(HUNTER_ENABLED)
	find_package(PythonInterp 3)
	if(PYTHONINTERP_FOUND)
		hunter_gate_self(
			"${HUNTER_CACHED_ROOT}"
			"${HUNTER_VERSION}"
			"${HUNTER_SHA1}"
			hunter_dir
		)
		set(hunter_upload_script "${hunter_dir}/maintenance/upload-cache-to-github.py")
		set(hunter_cache_dir "${HUNTER_GATE_ROOT}/_Base/Cache")
		set(hunter_tmp_dir "${HUNTER_GATE_ROOT}/tmp")
		add_custom_target(
			hunter_upload_cache
			${PYTHON_EXECUTABLE} ${hunter_upload_script}
			--username hunter-cache-bot
			--repo-owner ethereum
			--repo hunter-cache
			--cache-dir ${hunter_cache_dir}
			--temp-dir ${hunter_tmp_dir}
		)
	endif()
endif()


project(miner)
set(PROJECT_VERSION JCEMiner.1.0.6)

cable_set_build_type(DEFAULT Release CONFIGURATION_TYPES Release RelWithDebInfo Debug)

# link_directories interprate relative paths with respect to CMAKE_CURRENT_SOURCE_DIR
cmake_policy(SET CMP0015 NEW)

# let cmake autolink dependencies on windows
# it's specified globally, cause qt libraries requires that on windows and they are also found globally
cmake_policy(SET CMP0020 NEW)

# 3.1 and above
cmake_policy(SET CMP0054 NEW)

######################################################################################################

option(ETHASHCL "Build with OpenCL mining" ON)
option(ETHASHCUDA "Build with CUDA mining" ON)
option(APICORE "Build with API Server support" ON)

# propagates CMake configuration options to the compiler
function(configureProject)
	if (ETHASHCL)
		add_definitions(-DETH_ETHASHCL)
	endif()
	if (ETHASHCUDA)
		add_definitions(-DETH_ETHASHCUDA)
	endif()
        if (APICORE)
                add_definitions(-DAPI_CORE)
        endif()
endfunction()

hunter_add_package(Boost COMPONENTS system program_options)
find_package(Boost CONFIG REQUIRED COMPONENTS system program_options)

hunter_add_package(jsoncpp)
find_package(jsoncpp CONFIG REQUIRED)

hunter_add_package(libjson-rpc-cpp)
find_package(libjson-rpc-cpp CONFIG REQUIRED)

hunter_add_package(CppNetlibUri)
find_package(CppNetlibUri CONFIG REQUIRED)

configureProject()

message("------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                  ${ETHASHCL}")
message("-- ETHASHCUDA       Build CUDA components                    ${ETHASHCUDA}")
message("-- APICORE          Build API Server components              ${APICORE}")
message("------------------------------------------------------------------------")
message("")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(EthCompilerSettings)

cable_add_buildinfo_library(PREFIX miner)

add_subdirectory(libdevcore)
add_subdirectory(libethash)
add_subdirectory(libethcore)
add_subdirectory(libhwmon)
add_subdirectory(libproto)

if (ETHASHCL)
	add_subdirectory(libcl)
endif ()
if (ETHASHCUDA)
	add_subdirectory(libcuda)
endif ()
if (APICORE)
       add_subdirectory(libapi)
endif ()

add_subdirectory(miner)


set(CPACK_GENERATOR ZIP)
set(CPACK_PACKAGE_FILE_NAME ${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
include(CPack)


# Export some variables for CIs
set(PROJECT_VERSION_IS_PRERELEASE false)
configure_file(cmake/variables.sh.in variables.sh)
