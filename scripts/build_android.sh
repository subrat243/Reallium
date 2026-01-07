#!/bin/bash

# Build script for Edge Agent (Android)

set -e

echo "Building Deepfake Edge Agent for Android..."

# Configuration
BUILD_DIR="build/android"
NDK_PATH="${ANDROID_NDK_HOME:-$ANDROID_NDK}"
ABI="arm64-v8a"  # Target ARM64
API_LEVEL=26

if [ -z "$NDK_PATH" ]; then
    echo "Error: ANDROID_NDK_HOME not set"
    exit 1
fi

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
cmake ../.. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ABI \
    -DANDROID_PLATFORM=android-$API_LEVEL \
    -DBUILD_ANDROID=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release -j$(nproc)

echo "Build complete!"
echo "Output: $BUILD_DIR"
