set(SOURCES
    api/Api.h api/Api.cpp api/ApiServer.h api/ApiServer.cpp
)

add_library(api ${SOURCES})
target_link_libraries(api PRIVATE devcore miner-buildinfo libjson-rpc-cpp::server)
target_include_directories(api PRIVATE ..)
