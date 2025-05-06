#!/bin/bash

export PY_BIN="$CONDA_PREFIX/bin/python"
cmake . \
    -DBUILD_PYTHON=ON \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DPython_EXECUTABLE="${CONDA_PREFIX}/bin/python" \
    -DCMAKE_CXX_FLAGS="-std=c++17" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
make -j 4
make install
