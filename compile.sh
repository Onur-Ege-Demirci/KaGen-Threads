#!/bin/bash
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKAGEN_BUILD_TESTS=ON 
cmake --build build --parallel --verbose 

