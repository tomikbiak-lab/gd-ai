cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

project(GDAI VERSION 1.0.0)

if(NOT WIN32)
    message(FATAL_ERROR "GD AI currently supports Windows only (the bridge uses WinSock).")
endif()

add_library(${PROJECT_NAME} SHARED
    src/main.cpp
    src/menu.cpp
    src/net.cpp
    src/replay.cpp
    src/stats.cpp
)

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ws2_32)
endif()

if (NOT DEFINED ENV{GEODE_SDK})
    message(FATAL_ERROR "Unable to find Geode SDK! Please define GEODE_SDK environment variable to point to Geode")
else()
    message(STATUS "Found Geode: $ENV{GEODE_SDK}")
endif()

add_subdirectory("$ENV{GEODE_SDK}" "${CMAKE_CURRENT_BINARY_DIR}/geode")

setup_geode_mod(${PROJECT_NAME} DONT_INSTALL)

# Geode's precompiled header pulls in windows.h, which clashes with winsock2.h.
set_source_files_properties(src/net.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
