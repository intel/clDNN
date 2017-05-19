# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "${CLDNN_BUILD__PROJ__clDNN}")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ_LABEL__clDNN}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ_OUTPUT_NAME__clDNN}")

# ================================== Compiler preprocessor definitions =================================

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    CLDNN_EXPORTS
    EXPORT_NEURAL_SYMBOLS
  )

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__api                 "api")
file(GLOB __CLDNN_Headers__api
    "${CLDNN__API_DIR}/*.h"
    "${CLDNN__API_DIR}/*.hpp"
  )

set(__CLDNN_Directory__api__cpp "${CLDNN__API_DIR}/CPP")
set(__CLDNN_Label__api__cpp     "${__CLDNN_Label__api}\\CPP")
file(GLOB __CLDNN_Headers__api__cpp
    "${__CLDNN_Directory__api__cpp}/*.h"
    "${__CLDNN_Directory__api__cpp}/*.hpp"
  )

set(__CLDNN_Directory__api__c "${CLDNN__API_DIR}/C")
set(__CLDNN_Label__api__c     "${__CLDNN_Label__api}\\C")
file(GLOB __CLDNN_Headers__api__c
    "${__CLDNN_Directory__api__c}/*.h"
    "${__CLDNN_Directory__api__c}/*.hpp"
  )

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

set(__CLDNN_Directory__include         "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(__CLDNN_Label__include             "include")
file(GLOB __CLDNN_Headers__include
    "${__CLDNN_Directory__include}/*.h"
    "${__CLDNN_Directory__include}/*.hpp"
  )

set(__CLDNN_Directory__gpu             "${CMAKE_CURRENT_SOURCE_DIR}/gpu")
set(__CLDNN_Label__gpu                 "gpu")
file(GLOB __CLDNN_Sources__gpu
    "${__CLDNN_Directory__gpu}/*.h"
    "${__CLDNN_Directory__gpu}/*.hpp"
    "${__CLDNN_Directory__gpu}/*.cpp"
    "${__CLDNN_Directory__gpu}/*.inc"
  )

set(__CLDNN_Directory__cache           "${__CLDNN_Directory__gpu}/cache")
set(__CLDNN_Label__cache               "${__CLDNN_Label__gpu}\\cache")
file(GLOB __CLDNN_Sources__cache
    "${__CLDNN_Directory__cache}/*.h"
    "${__CLDNN_Directory__cache}/*.hpp"
    "${__CLDNN_Directory__cache}/*.cpp"
  )

set(__CLDNN_Directory__ch_kernels      "${__CLDNN_Directory__cache}/kernels")
set(__CLDNN_Label__ch_kernels          "${__CLDNN_Label__cache}\\kernels")
file(GLOB __CLDNN_Sources__ch_kernels
    "${__CLDNN_Directory__ch_kernels}/*.cl"
  )

set(__CLDNN_Directory__cg_cache        "${CLDNN__CODEGEN_INCDIR}")
set(__CLDNN_CGDirectory__cg_cache      "${CLDNN__CODEGEN_DIR}/cache")
set(__CLDNN_Label__cg_cache            "${__CLDNN_Label__cache}\\codegen")
set(__CLDNN_File__cg_cache__prim_db    "primitive_db.inc")
set(__CLDNN_Sources__cg_cache
    "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
  )


set(__CLDNN_AllSources
    ${__CLDNN_Headers__api}
    ${__CLDNN_Headers__include}
    ${__CLDNN_Headers__api__cpp}
    ${__CLDNN_Headers__api__c}
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__gpu}
    ${__CLDNN_Sources__cache}
    ${__CLDNN_Sources__ch_kernels}
    ${__CLDNN_Sources__cg_cache}
  )
# Helping with some generators.
set_property(SOURCE ${__CLDNN_Sources__cg_cache} PROPERTY GENERATED TRUE)

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__api}"        FILES ${__CLDNN_Headers__api})
source_group("${__CLDNN_Label__api__cpp}"   FILES ${__CLDNN_Headers__api__cpp})
source_group("${__CLDNN_Label__api__c}"     FILES ${__CLDNN_Headers__api__c})
source_group("${__CLDNN_Label__include}"    FILES ${__CLDNN_Headers__include})
source_group("${__CLDNN_Label__main}"       FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__gpu}"        FILES ${__CLDNN_Sources__gpu})
source_group("${__CLDNN_Label__cache}"      FILES ${__CLDNN_Sources__cache})
source_group("${__CLDNN_Label__ch_kernels}" FILES ${__CLDNN_Sources__ch_kernels})
source_group("${__CLDNN_Label__cg_cache}"   FILES ${__CLDNN_Sources__cg_cache})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${__CLDNN_Directory__include}"
  )

# =================================== Link targets and dependencies ====================================

# Main shared library.
add_library("${CLDNN_BUILD__PROJ}" SHARED
    ${__CLDNN_AllSources}
  )
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")


target_link_libraries("${CLDNN_BUILD__PROJ}"
    OpenCL
    Boost::filesystem
    Boost::system
  )

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" setupapi)
elseif(UNIX)
  target_link_libraries("${CLDNN_BUILD__PROJ}" pthread)
endif()

# =================================== Custom pre- and post-steps =======================================

add_custom_command(OUTPUT "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${__CLDNN_CGDirectory__cg_cache}"
    COMMAND "${PYTHON_EXECUTABLE}" "${__CLDNN_Directory__cache}/primitive_db_gen.py" "${__CLDNN_CGDirectory__cg_cache}" "${__CLDNN_Directory__ch_kernels}"
    DEPENDS ${__CLDNN_Sources__ch_kernels} "${__CLDNN_Directory__cache}/primitive_db_gen.py"
    COMMENT "Generating ${__CLDNN_File__cg_cache__prim_db} ..."
  )
add_custom_command(OUTPUT "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}" "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    DEPENDS "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}" ${__CLDNN_Sources__ch_kernels} "${__CLDNN_Directory__cache}/primitive_db_gen.py"
    COMMENT "Updating file if the file changed (${__CLDNN_File__cg_cache__prim_db}) ..."
  )

# ========================================== Installation ==============================================

# API headers.
install(DIRECTORY "${CLDNN__API_DIR}/"
    DESTINATION "include/clDNN"
    FILE_PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
  )
# Binaries.
install(TARGETS "${CLDNN_BUILD__PROJ}"
    ARCHIVE DESTINATION "lib"
    LIBRARY DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "include/clDNN"
  )

# ======================================================================================================
