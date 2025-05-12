cmake . -G "NMake Makefiles" ^
    -DBUILD_PYTHON=ON ^
    -DCMAKE_PREFIX_PATH:PATH="%CONDA_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX:PATH="%CONDA_PREFIX%" ^
    -DPython_EXECUTABLE:PATH="%CONDA_PREFIX%\python.exe"
cmake --build . --target INSTALL --config Release -j 4
