cmake_minimum_required(VERSION 3.21)

# set the name of the projects
project(wolf
    DESCRIPTION "Wolf Engine"
    LANGUAGES CXX
)

set(TEST_PROJECT_NAME "${PROJECT_NAME}_tests")
message("CXX Compiler ID is ${CMAKE_CXX_COMPILER_ID}")

# set the options and enviroment variables
#set(WEBRTC_SRC $ENV{WEBRTC_ROOT} CACHE STRING "path to the root folder of webrtc folder")

#define includes, libs and srcs
set(INCLUDES)
set(LIBS PARENT_SCOPE) 
set(SRCS)
set(TESTS_SRCS)

# check the OS
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    if (WIN32)
        set(WIN64 TRUE)
    endif()
endif()

if(UNIX AND (NOT APPLE) AND (NOT EMSCRIPTEN))
   set(LINUX TRUE)
endif()

if (MSVC AND NOT WIN64)
    message( FATAL_ERROR "Only Window 64 bit is supported" )
endif()

if(ANDROID)
    set(MOBILE ON)
    message(STATUS "target platform is of mobile type")
elseif(WIN32 OR UNIX AND NOT EMSCRIPTEN)
    set(DESKTOP ON)
    message(STATUS "target platform is of desktop type")
endif()

# set target os
if (WIN64)
    set(TARGET_OS "win")
    set(LIB_EXT "lib")
    set(SHARED_EXT "dll")
elseif(APPLE)
    set(TARGET_OS "mac")
    set(LIB_EXT "a")
    set(SHARED_EXT "dylib")
elseif(LINUX)
    set(TARGET_OS "linux")
    set(LIB_EXT "a")
    set(SHARED_EXT "so")
elseif (NOT EMSCRIPTEN)
    message( FATAL_ERROR "Unsuported OS, please open an issue at https://github.com/WolfEngine/wolf" )
endif()

# required packages
find_package(Git REQUIRED)
if (NOT EMSCRIPTEN)
    find_package (Threads)
endif()

# use folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# FetchContent for cloning repositories, avaiable since CMAKE 3.11
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# set type of library
if (LINUX OR ANDROID)
    set(LIBRARY_TYPE "SHARED")
else()
    set(LIBRARY_TYPE "STATIC")
endif()

set(LIBRARY_TYPE "${LIBRARY_TYPE}" CACHE STRING "Library type")

if (CMAKE_BUILD_TYPE STREQUAL "" OR NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not specified. Defaulting to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# Build options
set(LIB_INSTALL_DIR lib CACHE STRING "Install location of libraries")
set(BIN_INSTALL_DIR bin CACHE STRING "Install location of executables")
set(INCLUDE_INSTALL_DIR include CACHE STRING "Install location of executables")

# CMAKE GUI Options
set(EMSDK "$ENV{EMSDK}" CACHE STRING "Emscripten SDK path")
set(BOOST_VERSION "1.82.0" CACHE STRING "Boost version tag")

# helper macro to define feature option
# with conditional default value since
# cmake doesn't support returning value
# or boolean expression outside `if()`.
macro(feature_option FEATURE_NAME DESCRIPTION)
    if (${ARGN})
        set(DEFAULT_VALUE ON)
    else()
        set(DEFAULT_VALUE OFF)
    endif()
    option(${FEATURE_NAME} ${DESCRIPTION} ${DEFAULT_VALUE})
endmacro()

# features sets (select a set of features with a single option)
option(WOLF_FEATURES_ALL "Enable all features by default" OFF)

# media modules
feature_option(WOLF_MEDIA_FFMPEG "Enable ffmpeg encoding and decoding" WOLF_FEATURES_ALL AND (WIN32 OR ANDROID) AND NOT EMSCRIPTEN)
feature_option(WOLF_MEDIA_OPENAL "Enable OpenAL soft" WOLF_FEATURES_ALL AND NOT LINUX AND NOT EMSCRIPTEN)
feature_option(WOLF_MEDIA_SCREEN_CAPTURE "Enable screen capture" WOLF_FEATURES_ALL AND DESKTOP)
feature_option(WOLF_MEDIA_STB "Enable stb headers" WOLF_FEATURES_ALL)
feature_option(WOLF_MEDIA_GSTREAMER "Enable gstreamer wrapper" OFF)

# stream modules
feature_option(WOLF_STREAM_GRPC   "Enable gRPC connection" OFF)
feature_option(WOLF_STREAM_QUIC   "Enable QUIC" OFF)
feature_option(WOLF_STREAM_RIST   "Enable RIST streaming protocol" OFF)
feature_option(WOLF_STREAM_WEBRTC "Enable webRTC" OFF)

# system modules
feature_option(WOLF_SYSTEM_GAMEPAD_CLIENT "Enable gamepad input handling" WOLF_FEATURES_ALL AND NOT LINUX AND NOT EMSCRIPTEN)
feature_option(WOLF_SYSTEM_GAMEPAD_VIRTUAL "Enable virtual gamepad simulator" OFF)
feature_option(WOLF_SYSTEM_HTTP_WS "Enable http1.1 and websocket client/server based on boost beast or Emscripten" WOLF_FEATURES_ALL)
feature_option(WOLF_SYSTEM_LOG "Enable log" WOLF_FEATURES_ALL AND NOT EMSCRIPTEN)
feature_option(WOLF_SYSTEM_LZ4 "Enable lz4 for compression" WOLF_FEATURES_ALL AND NOT EMSCRIPTEN)
feature_option(WOLF_SYSTEM_LZMA "Enable lzma for compression" WOLF_FEATURES_ALL AND DESKTOP)
feature_option(WOLF_SYSTEM_LUA "Enable lua scripting language" WOLF_FEATURES_ALL AND DESKTOP)
feature_option(WOLF_SYSTEM_MIMALLOC "Enable Microsoft's mimalloc memory allocator" WOLF_FEATURES_ALL AND DESKTOP)
feature_option(WOLF_SYSTEM_POSTGRESQL "Enable postgresql database client" WOLF_FEATURES_ALL AND DESKTOP)
feature_option(WOLF_SYSTEM_PYTHON "Enable embedded Python3 scripting" WOLF_FEATURES_ALL AND DESKTOP AND NOT LINUX)
feature_option(WOLF_SYSTEM_REDIS "Enable redis client based on hiredis" OFF)
feature_option(WOLF_SYSTEM_SIG_SLOT "Enable signal/slot based on boost signals2" WOLF_FEATURES_ALL)
feature_option(WOLF_SYSTEM_SOCKET "Enable TCP/UDP protocol over socket" WOLF_FEATURES_ALL AND NOT ANDROID)
feature_option(WOLF_SYSTEM_OPENSSL "Enable openSSL" WOLF_FEATURES_ALL)
feature_option(WOLF_SYSTEM_STACKTRACE "Enable boost stacktrace" WOLF_FEATURES_ALL AND WIN32 AND NOT EMSCRIPTEN)
feature_option(WOLF_SYSTEM_ZLIB "Enable Zlib compression library" WOLF_FEATURES_ALL AND NOT EMSCRIPTEN)

# machine learing modules
feature_option(WOLF_ML_NUDITY_DETECTION "Enable machine learning nudity detection" OFF)
feature_option(WOLF_ML_OCR "Enable machine learning referee ocr" OFF)

#option(WOLF_ENABLE_LTO "Enable cross language linking time optimization" OFF)
option(WOLF_TEST "Enable tests" ON)
if (NOT MSVC)
    option(WOLF_ENABLE_ASAN "Enable ASAN" OFF)
endif()

if(ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT supported OUTPUT error)
  if(supported)
      message(STATUS "IPO / LTO enabled")
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
      add_link_options(-fuse-ld=lld)
  else()
      message(STATUS "IPO / LTO not supported: <${error}>")
  endif()
endif()

# set C/CXX standards
set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# message("DDDDD ${CMAKE_BUILD_TYPE}")
# if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
#   set(CMAKE_BUILD_TYPE "Debug")
# endif()

# set C++ flags based on compiler
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  # using Clang or AppleClang
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2b -fexceptions -fcoroutines-ts")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2b -fexceptions -fcoroutines")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Microsoft Visual C++
  # set C++23 as the primary standard
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_FLAGS "/EHsc /W3 /bigobj")
endif()

set(GETOPT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/system)

# complex/pre-built wolf dependencies to be made available.
set(WDEPS_MSQUIC ${WOLF_STREAM_QUIC})
set(WDEPS_FFMPEG5 ${WOLF_MEDIA_FFMPEG})
FetchContent_Declare(
    wolf-deps
    GIT_REPOSITORY https://github.com/WolfEngine/wolf-deps
    GIT_TAG        main
    GIT_SHALLOW    ON
)
FetchContent_MakeAvailable(wolf-deps)

# include cmakes
include(cmake/vcpkg.cmake)
include(cmake/system.cmake)
include(cmake/stream.cmake)
include(cmake/media.cmake)
include(cmake/ml.cmake)

if (EMSCRIPTEN)
    message(WARNING "WOLF_TEST will be disabled for Emscripten")
    set(WOLF_TEST OFF)
else()
    # currently threads was not supported with WASM
    list(APPEND LIBS Threads::Threads)
endif()

# disable build testing
set(BUILD_TESTING OFF CACHE BOOL "BUILD_TESTING")

if (WOLF_ENABLE_ASAN)
    set(ENABLE_ASAN TRUE)
endif()

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

# enabling clang-tidy
# can be enabled with .CLANG-TIDY from Visual Studio Code
# https://devblogs.microsoft.com/cppblog/visual-studio-code-c-december-2021-update-clang-tidy/
# can be enabled with .CLANG-TIDY from Visual Studio
# https://devblogs.microsoft.com/cppblog/code-analysis-with-clang-tidy-in-visual-studio/
#set(CMAKE_C_CLANG_TIDY
#     clang-tidy;
#     -format-style=file;) 
#set(CMAKE_CXX_CLANG_TIDY
#     clang-tidy;
#     -format-style=file;)  

# include sources
file(GLOB_RECURSE WOLF_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/wolf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wolf.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DISABLE_ANALYSIS_BEGIN"
    "${CMAKE_CURRENT_SOURCE_DIR}/DISABLE_ANALYSIS_END"
)

file(GLOB_RECURSE WOLF_PROTOS
    "${CMAKE_CURRENT_SOURCE_DIR}/protos/*"
)

file(GLOB_RECURSE WOLF_CMAKES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*"
)

# add source codes
add_library(${PROJECT_NAME} ${LIBRARY_TYPE} 
    ${SRCS}
    ${WOLF_SRCS}
    ${WOLF_CMAKES}
    ${WOLF_PROTOS}
)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${INCLUDES}
	${CMAKE_CURRENT_SOURCE_DIR} 
	${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS})

if (WOLF_STREAM_RIST)
    add_dependencies(${PROJECT_NAME} ${RIST_TARGET})
endif()
		
if (MSVC OR WIN32)
    if (LIBRARY_TYPE STREQUAL "STATIC")
        set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DDEBUG)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DNDEBUG)
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC 
    -DBOOST_ASIO_NO_DEPRECATED
    -DBOOST_ASIO_HAS_CO_AWAIT
    -DBOOST_ASIO_HAS_STD_COROUTINE 
)
if (MSVC)
    target_compile_definitions(${PROJECT_NAME} PUBLIC 
        -EHsc
        -DNOMINMAX
        -DWIN32_LEAN_AND_MEAN
        -DWIN64
        -DWIN32
        -D_WIN32_WINNT=0x0601 # Windows 7
    )
elseif (EMSCRIPTEN)
    target_compile_definitions(${PROJECT_NAME} PUBLIC 
        -DEMSCRIPTEN
    )
elseif(APPLE)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DNEED_XLOCALE_H=1)
endif()

# setup Wolf definitions
get_cmake_property(_vars VARIABLES)
foreach (_var ${_vars})
    string(FIND ${_var} "WOLF_" out)
    if(("${out}" EQUAL 0) AND ("(${${_var}}" MATCHES ON))
        target_compile_definitions(${PROJECT_NAME} PUBLIC "-D${_var}")
    endif()
endforeach()

# create source group
source_group("wolf" FILES ${WOLF_SRCS})
source_group("cmake" FILES ${WOLF_CMAKES})
source_group("protos" FILES ${WOLF_PROTOS})
source_group("stream/grpc" FILES ${WOLF_STREAM_GRPC_SRC})
source_group("stream/janus" FILES ${WOLF_STREAM_JANUS_SRC})
source_group("stream/test" FILES ${WOLF_STREAM_TEST_SRC})
source_group("stream/quic/datatypes" FILES ${WOLF_STREAM_QUIC_DATATYPES_SRC})
source_group("stream/quic/events" FILES ${WOLF_STREAM_QUIC_EVENTS_SRC})
source_group("stream/quic/handles" FILES ${WOLF_STREAM_QUIC_HANDLES_SRC})
source_group("stream/quic/internal" FILES ${WOLF_STREAM_QUIC_INTERNAL_SRC})
source_group("stream/quic" FILES ${WOLF_STREAM_QUIC_SRC})
source_group("stream/rist" FILES ${WOLF_STREAM_RIST_SRC})
source_group("stream/webrtc/capturer" FILES ${WOLF_STREAM_WEBRTC_CAPTURER_SRC})
source_group("stream/webrtc/data" FILES ${WOLF_STREAM_WEBRTC_DATA_SRC})
source_group("stream/webrtc/interceptor" FILES ${WOLF_STREAM_WEBRTC_INTERCEPTOR_SRC})
source_group("stream/webrtc/media" FILES ${WOLF_STREAM_WEBRTC_MEDIA_SRC})
source_group("stream/webrtc/peer" FILES ${WOLF_STREAM_WEBRTC_PEER_SRC})
source_group("stream" FILES ${WOLF_STREAM_SRC})
source_group("system/db" FILES ${WOLF_SYSTEM_REDIS_SRC})
source_group("system/gamepad" FILES ${WOLF_SYSTEM_GAMEPAD_CLIENT_SRC} ${WOLF_SYSTEM_GAMEPAD_VIRTUAL_SRCS})
source_group("system/log" FILES ${WOLF_SYSTEM_LOG_SRC})
source_group("system/compression" FILES ${WOLF_SYSTEM_LZ4_SRCS} ${WOLF_SYSTEM_LZMA_SRCS})
source_group("system/script" FILES ${WOLF_SYSTEM_LUA_SRC} ${WOLF_SYSTEM_PYTHON_SRC})
source_group("system/socket" FILES ${WOLF_SYSTEM_SOCKET_SRC} ${WOLF_SYSTEM_HTTP_WS_SRC})
source_group("system/test" FILES ${WOLF_SYSTEM_TEST_SRC})
source_group("system" FILES ${WOLF_SYSTEM_SRC})
source_group("media/test" FILES ${WOLF_MEDIA_TEST_SRC})
source_group("media/ffmpeg" FILES ${WOLF_MEDIA_FFMPEG_SRC})
source_group("media" FILES ${WOLF_MEDIA_OPENAL_SRC} ${WOLF_MEDIA_STB_SRC})
source_group("ml/referee_ocr" FILES ${WOLF_ML_OCR_SRC})
source_group("ml/nudity_detection" FILES ${WOLF_ML_NUD_DET_SRC})

# add compile options
if (NOT WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE -std=c++2b -fPIC)
endif()

if (WOLF_ENABLE_ASAN)
    target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
    target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
endif()

# build tests
if (WOLF_TEST)
    add_executable(${TEST_PROJECT_NAME}
        tests.cpp
        ${TESTS_SRCS}
    )

    if (MSVC OR WIN32)
        if (LIBRARY_TYPE STREQUAL "STATIC")
            set_property(TARGET ${TEST_PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        else()
            set_property(TARGET ${TEST_PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        endif()
    endif()

    if(WOLF_ML_OCR AND LINUX)
        target_link_libraries(${TEST_PROJECT_NAME} PRIVATE ${PROJECT_NAME} ${leptonica_BINARY_DIR}/install/lib/libleptonica.so)
    else()
        target_link_libraries(${TEST_PROJECT_NAME} PRIVATE ${PROJECT_NAME})
    endif()

    if (NOT WIN32)
        target_compile_options(${TEST_PROJECT_NAME} PRIVATE -std=c++2b)
    endif()

    include(CTest)
    add_test(NAME ${TEST_PROJECT_NAME} COMMAND ${TEST_PROJECT_NAME})

    set_tests_properties(${TEST_PROJECT_NAME}
        PROPERTIES
            ENVIRONMENT WOLF_CONTENT_PATH="${CMAKE_SOURCE_DIR}/content"
    )

    # copy runtime dll files to the same directory as the executable.
    if(WIN32)
        add_custom_command(TARGET ${TEST_PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TEST_PROJECT_NAME}> $<TARGET_RUNTIME_DLLS:${TEST_PROJECT_NAME}> $<TARGET_FILE_DIR:${TEST_PROJECT_NAME}>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

if( WOLF_ML_NUDITY_DETECTION AND WIN64)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${TEST_PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:${TEST_PROJECT_NAME}>)
endif()

if((WOLF_ML_OCR OR WOLF_ML_NUDITY_DETECTION) AND WIN64)
    add_custom_command(TARGET ${TEST_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE} ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${CMAKE_BUILD_TYPE}
    )
endif()

# install target
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
	
foreach(ITEM ${INCLUDES})
  install(DIRECTORY ${ITEM}/ DESTINATION ${INCLUDE_INSTALL_DIR})
endforeach()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/wolf.hpp DESTINATION ${INCLUDE_INSTALL_DIR})

if (WOLF_TEST)
	install(TARGETS ${TEST_PROJECT_NAME} DESTINATION ${BIN_INSTALL_DIR})
endif()
