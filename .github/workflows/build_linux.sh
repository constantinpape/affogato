#!/bin/bash

export PY_BIN="$CONDA_PREFIX/bin/python"
cmake . \
    -DBUILD_PYTHON=ON \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DPYTHON_EXECUTABLE="$PY_BIN" \
    -DCMAKE_CXX_FLAGS="-std=c++17" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DPYTHON_MODULE_INSTALL_DIR="${CONDA_PREFIX}/lib/python${PY_VER}"
make -j 4
make install
