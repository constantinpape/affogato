cmake . -G "NMake Makefiles" ^
    -DBUILD_PYTHON=ON ^
    -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX="%CONDA_PREFIX%" ^
    -DPYTHON_EXECUTABLE="%PYTHON%"
cmake --build . --target INSTALL --config Release -j 4
