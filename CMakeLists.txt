#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

cmake_minimum_required(VERSION 3.16)
include(FetchContent)

project(intel_npu_acceleration_library)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # lots of warnings and all warnings as errors
    message(STATUS "Setting GCC/Clang specific flags for the entire build")
    add_compile_options(-Wall -Wextra -Werror -pedantic -Wdouble-promotion -Wfloat-conversion -march=native)
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(STATUS "Setting Visual Studio specific flags for the entire build")
    add_compile_options(/W3 /WX /arch:AVX2 /arch:SSE2)
    add_link_options(/WX)
else()
    message(AUTHOR_WARNING "-- Building with unrecognised compiler, not setting any specific flags")
endif()

function(get_linux_lsb_release_information)
    find_program(LSB_RELEASE_CMD lsb_release)
    if(NOT LSB_RELEASE_CMD)
        message(FATAL_ERROR "Command lsb_release cannot be found")
    endif()

    execute_process(COMMAND "${LSB_RELEASE_CMD}" --short --id OUTPUT_VARIABLE LSB_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${LSB_RELEASE_CMD}" --short --release OUTPUT_VARIABLE LSB_RELEASE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(LSB_RELEASE_ID "${LSB_RELEASE_ID}" PARENT_SCOPE)
    set(LSB_RELEASE_VERSION "${LSB_RELEASE_VERSION}" PARENT_SCOPE)
endfunction()

if (WIN32)
    if(NOT OV_LIBRARY_URL)
        set(OV_LIBRARY_URL "https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/windows/w_openvino_toolkit_windows_2024.0.0.14509.34caeefd078_x86_64.zip")
    endif()
elseif(UNIX)
    if(NOT OV_LIBRARY_URL)
        get_linux_lsb_release_information()
        if (LSB_RELEASE_ID STREQUAL "Ubuntu")
            if (${LSB_RELEASE_VERSION} STREQUAL "18.04" OR ${LSB_RELEASE_VERSION} STREQUAL "20.04" OR ${LSB_RELEASE_VERSION} STREQUAL "22.04")
                string(REPLACE ".04" "" LSB_RELEASE_VERSION_SHORT ${LSB_RELEASE_VERSION})
                set(OV_LIBRARY_URL "https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/linux/l_openvino_toolkit_ubuntu${LSB_RELEASE_VERSION_SHORT}_2024.0.0.14509.34caeefd078_x86_64.tgz")
            else()
                message(FATAL_ERROR "Ubuntu version ${LSB_RELEASE_VERSION} is unsupported")
            endif()
        else()
            message(FATAL_ERROR "Linux distribution ${LSB_RELEASE_ID} is unsupported")
        endif()

    endif()
else()
    message(FATAL_ERROR "Unsupported architecture")
endif ()


FetchContent_Declare(
    openvino
    URL ${OV_LIBRARY_URL}
    )
FetchContent_MakeAvailable(openvino)

find_package(OpenVINO REQUIRED PATHS ${openvino_SOURCE_DIR}/runtime/cmake)

if (WIN32)
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    file(GLOB OpenVINObin ${openvino_SOURCE_DIR}/runtime/bin/intel64/Release/*)
    file(COPY ${OpenVINObin} DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release)

    file(GLOB TBBlib ${openvino_SOURCE_DIR}/runtime/3rdparty/tbb/bin/*)
    file(COPY ${TBBlib} DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release)
else()
    file(GLOB OpenVINObin ${openvino_SOURCE_DIR}/runtime/lib/intel64/*)
    file(COPY ${OpenVINObin} DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

# Create the Python module
add_library(intel_npu_acceleration_library SHARED src/bindings.cpp)

# Link the OpenVINO libraries
target_link_libraries(intel_npu_acceleration_library PRIVATE openvino::runtime)
if (UNIX)
    set_target_properties(intel_npu_acceleration_library PROPERTIES LINK_FLAGS "-Wl,-rpath,./")
endif (UNIX)


