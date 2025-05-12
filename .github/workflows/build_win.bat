for /f "usebackq delims=" %%I in (`python -c "import numpy; print(numpy.get_include())"`) do (
    set "NUMPY_INCLUDE_DIR=%%I"
)

cmake . -G "NMake Makefiles" ^
    -DBUILD_PYTHON=ON ^
    -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX="%CONDA_PREFIX%" ^
    -DPYTHON_EXECUTABLE="%PYTHON%" ^
    -DPython3_NumPy_INCLUDE_DIR=%NUMPY_INCLUDE_DIR% ^
    -DPython_NumPy_INCLUDE_DIR=%NUMPY_INCLUDE_DIR%
cmake --build . --target INSTALL --config Release -j 4
