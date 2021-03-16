#!/bin/bash

export PY_BIN="$CONDA_PREFIX/bin/python"
cmake . \
    # -DWITHIN_TRAVIS=ON \
    -DBUILD_PYTHON=ON \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DPYTHON_EXECUTABLE="$PY_BIN" \
    -DCMAKE_CXX_FLAGS="-std=c++17" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DBUILD_NIFTY_PYTHON=ON
make -j 4
make install
