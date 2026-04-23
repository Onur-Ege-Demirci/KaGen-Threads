#!/bin/bash
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKAGEN_BUILD_EXAMPLES=OFF -DKAGEN_BUILD_TOOLS=OFF
cmake --build build --parallel --verbose 

