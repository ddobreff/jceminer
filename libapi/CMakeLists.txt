set(SOURCES
    api/Api.h api/Api.cpp api/ApiServer.h api/ApiServer.cpp
)

hunter_add_package(mongoose)
find_package(mongoose CONFIG REQUIRED)

add_library(api ${SOURCES})
target_link_libraries(api PRIVATE devcore miner-buildinfo libjson-rpc-cpp::server mongoose::mongoose)
target_include_directories(api PRIVATE ..)
