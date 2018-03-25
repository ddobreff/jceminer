set(SOURCES
	PoolURI.cpp PoolURI.h
	PoolClient.h
	PoolManager.h PoolManager.cpp
	stratum/EthStratumClient.h stratum/EthStratumClient.cpp
)

hunter_add_package(OpenSSL)
find_package(OpenSSL REQUIRED)

hunter_add_package(thread-pool-cpp)
find_package(thread-pool-cpp CONFIG REQUIRED)

add_library(proto ${SOURCES})
target_link_libraries(proto PRIVATE devcore miner-buildinfo Boost::system jsoncpp_lib_static OpenSSL::SSL OpenSSL::Crypto network-uri thread-pool-cpp::thread-pool-cpp)
target_include_directories(proto PRIVATE ..)
