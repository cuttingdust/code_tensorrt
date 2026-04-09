cmake_minimum_required(VERSION 3.12)
if(POLICY CMP0177)
	cmake_policy(SET CMP0177 NEW)
endif()

# 使用 vcpkg 工具链文件 Warning  工具连设置必须是在project 前面
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# 获取父目录
get_filename_component(CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)  
get_filename_component(PARENT_DIR "${CURRENT_DIR}" DIRECTORY)  
get_filename_component(PARENT_NAME ${PARENT_DIR} NAME)
message(STATUS "Current Directory: ${CURRENT_DIR}")
message(STATUS "Parent Directory: ${PARENT_DIR}")
message(STATUS "Parent Directory Name: ${PARENT_NAME}")

project(${PARENT_NAME})
message("PROJECT_NAME: ${PROJECT_NAME}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
if(MSVC)
set(CMAKE_GENERATOR_PLATFORM x64)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

# OpenMP
find_package(OpenMP)
message("OpenMP_FOUND = ${OpenMP_FOUND}")

if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# #MFC
# find_package(MFC)
# message("MFC_FOUND = ${MFC_FOUND}")

# if(MFC_FOUND)
# #1: static MFC library, 2: shared MFC library
# set(CMAKE_MFC_FLAG 2)
# endif()

# #Python3
# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# message("Python3_FOUND = ${Python3_FOUND}")
# message("Python3_VERSION = ${Python3_VERSION}")
# message("Python3_LIBS = ${Python3_LIBRARIES}")
#==================================================================================#

# OpenCV
find_package(OpenCV CONFIG REQUIRED)
message("OpenCV_FOUND = ${OpenCV_FOUND}")
message("OpenCV_VERSION = ${OpenCV_VERSION}")
message("OpenCV_LIBS = ${OpenCV_LIBS}")

# 外部依赖项
set(DPS_INCLUDES)
set(DPS_TARGETS)
set(DPS_LIBRARYS)

# 资源文件

# #图标
set(RC_FILE)

# #qrc 文件
set(QRC_SOURCE_FILES)

# 忽略特定库
if(MSVC)
SET(CMAKE_EXE_LINKER_FLAGS /NODEFAULTLIB:"libcmt.lib;libc.lib;msvcrt.lib")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CommonTools)

add_subdirectory(memory_array_opencv)

# 拷贝assert目录到输出目录
set(ASSERT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/assert)
file(COPY ${ASSERT_PATH} DESTINATION ${OUT_RUN_PATH})
