add_subdirectory(lib)


#################################
# INSTALL THE PYTHON MODULE
#################################

IF(NOT DEFINED PYTHON_MODULE_INSTALL_DIR OR PYTHON_MODULE_INSTALL_DIR MATCHES "^$")


    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c 
      "from distutils import sysconfig as sc; print(sc.get_python_lib(prefix='', plat_specific=True))"
      OUTPUT_VARIABLE PYTHON_SITE
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
      OUTPUT_VARIABLE PYTHON_SITE
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    SET(PYTHON_MODULE_INSTALL_DIR ${PYTHON_SITE})

ENDIF()

SET(PYTHON_MODULE_INSTALL_DIR ${PYTHON_MODULE_INSTALL_DIR}
    CACHE PATH "where to install the affogato python package" FORCE)

# this is the install path relative to CMAKE_INSTALL_PREFIX,
# use this in INSTALL() commands to get packaging right
FILE(RELATIVE_PATH PYTHON_MODULE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_MODULE_INSTALL_DIR})



install(DIRECTORY ${CMAKE_BINARY_DIR}/python/affogato
    DESTINATION ${PYTHON_MODULE_INSTALL_DIR})
