#!/bin/bash

# Build script for Edge Agent (iOS)

set -e

echo "Building Deepfake Edge Agent for iOS..."

# Configuration
BUILD_DIR="build/ios"
PLATFORM="OS64"  # iOS device
DEPLOYMENT_TARGET="12.0"

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
cmake ../.. \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=$DEPLOYMENT_TARGET \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DBUILD_IOS=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

echo "Build complete!"
echo "Output: $BUILD_DIR"
