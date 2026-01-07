cmake_minimum_required(VERSION 3.16)
project(DeepfakeEdgeAgent VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_ANDROID "Build for Android" OFF)
option(BUILD_IOS "Build for iOS" OFF)
option(BUILD_TESTS "Build tests" ON)

# TensorFlow Lite
find_package(TensorFlowLite REQUIRED)

# OpenSSL for security
find_package(OpenSSL REQUIRED)

# SQLite for local storage
find_package(SQLite3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORFLOW_LITE_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
    ${SQLITE3_INCLUDE_DIRS}
)

# Core library
add_library(deepfake_core STATIC
    core/inference_engine.cpp
    core/detector.cpp
    security/auth_manager.cpp
    security/firmware_verifier.cpp
    offline/cache_manager.cpp
)

target_link_libraries(deepfake_core
    ${TENSORFLOW_LITE_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    ${SQLITE3_LIBRARIES}
)

# Android specific
if(BUILD_ANDROID)
    add_subdirectory(mobile/android)
endif()

# iOS specific
if(BUILD_IOS)
    add_subdirectory(mobile/ios)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS deepfake_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY core/ security/ offline/
    DESTINATION include/deepfake
    FILES_MATCHING PATTERN "*.h"
)
