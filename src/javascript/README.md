# Compiling affogato in emscripten

Prototype:
- Exports 2d mutex watershed in `mws_functions.cpp`
- Example on how to use it in javascript in `test_mws.html`


## Set up emscripten

https://emscripten.org/docs/getting_started/downloads.html


## Configure & Build

**Configure**:
Create a build folder `<BUILD_DIR>`.
```
emcmake cmake /PATH/TO/AFFOGATO -DCMAKE_PREFIX_PATH=/PATH/TO/CONDA_ENV -DBUILD_PYTHON=OFF -DBUILD_JS=ON
```
If boost is not found, it needs to be set via ccmake (for some reason using `-DBoost_INCLUDE_DIR` doesn't work).

**Build**:
```
emmake make
```

## Run test 

Copy `mws.js` and `mws.wasm` from `<BUILD_DIR>/src/javascript` and `test_mws.html` into the same folder.
Serve the html via `python -m http.server`. Open in browser, select `test_mws.html` and open the console to check the output.
