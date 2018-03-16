cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

set(EXECUTABLE miner)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

target_link_libraries(miner PRIVATE ethcore ethash proto devcore miner-buildinfo Boost::program_options)

if(APICORE)
	target_link_libraries(miner PRIVATE apicore)
endif()

include(GNUInstallDirs)
install(TARGETS miner DESTINATION ${CMAKE_INSTALL_BINDIR})
