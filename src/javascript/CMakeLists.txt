add_executable(mws mws_functions.cpp)
# to avoid linker errors, see https://stackoverflow.com/questions/37747205/cmake-project-for-emscripten
set_target_properties(mws PROPERTIES LINK_FLAGS "-s DEMANGLE_SUPPORT=1 -s ALLOW_MEMORY_GROWTH=1 --bind")
