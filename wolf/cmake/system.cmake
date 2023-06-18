if (WOLF_SYSTEM_STACKTRACE)
    if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_SYSTEM_STACKTRACE")
    elseif(NOT MSVC)
        message(FATAL_ERROR "WOLF_SYSTEM_STACKTRACE is only supported on Visual C++")
    endif()
endif()


# fetch mimalloc
if (WOLF_SYSTEM_MIMALLOC)
    if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_SYSTEM_MIMALLOC")
    endif()
    vcpkg_install(mimalloc mimalloc TRUE)
    list(APPEND LIBS mimalloc)
endif()

# fetch boost components via vcpkg
if (EMSCRIPTEN)
    vcpkg_install_force(
        boost-leaf 
        boost-signals2
    )
elseif(WOLF_SYSTEM_PYTHON)
    vcpkg_install_force(  
        boost-asio 
        boost-beast 
        boost-leaf 
        boost-python
        boost-signals2 
        boost-test
    )
else()
    vcpkg_install_force(  
        boost-asio 
        boost-beast 
        boost-leaf 
        boost-signals2 
        boost-test
    )
endif()

set(Boost_INCLUDE_DIR $ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/include CACHE STRING "boost include directory" FORCE)
list(APPEND INCLUDES ${Boost_INCLUDE_DIR})
find_package(Boost ${Boost_VERSION} REQUIRED)

# install gsl
vcpkg_install(Microsoft.GSL ms-gsl FALSE)
list(APPEND LIBS Microsoft.GSL::GSL)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND NOT EMSCRIPTEN)
    vcpkg_install(fmt fmt FALSE)
    list(APPEND LIBS fmt::fmt-header-only)
endif()

if (WOLF_SYSTEM_GAMEPAD_CLIENT)
    file(GLOB_RECURSE WOLF_SYSTEM_GAMEPAD_CLIENT_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_client_emc.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_client_keymap.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_client_sdl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_client_types.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_client.hpp"
    )
    list(APPEND SRCS ${WOLF_SYSTEM_GAMEPAD_CLIENT_SRC})

    if (NOT EMSCRIPTEN)
        message("fetching https://github.com/libsdl-org/SDL")
        FetchContent_Declare(
            SDL3-static
            GIT_REPOSITORY https://github.com/libsdl-org/SDL
            GIT_TAG        main
        )

        set(SDL_AUDIO OFF CACHE BOOL "SDL_AUDIO")
        set(SDL_DIRECTX OFF CACHE BOOL "SDL_DIRECTX")
        set(SDL_DISKAUDIO OFF CACHE BOOL "SDL_DISKAUDIO")
        set(SDL_DUMMYAUDIO OFF CACHE BOOL "SDL_DUMMYAUDIO")
        set(SDL_DUMMYVIDEO OFF CACHE BOOL "SDL_DUMMYVIDEO")
        set(SDL_FILE OFF CACHE BOOL "SDL_FILE")
        set(SDL_FILESYSTEM OFF CACHE BOOL "SDL_FILESYSTEM")
        set(SDL_METAL OFF CACHE BOOL "SDL_METAL")
        set(SDL_OFFSCREEN OFF CACHE BOOL "SDL_OFFSCREEN")
        set(SDL_OPENGL OFF CACHE BOOL "SDL_OPENGL")
        set(SDL_OPENGLES OFF CACHE BOOL "SDL_OPENGLES")
        set(SDL_RENDER OFF CACHE BOOL "SDL_RENDER")
        set(SDL_RENDER_D3D OFF CACHE BOOL "SDL_RENDER_D3D")
        set(SDL_RENDER_METAL OFF CACHE BOOL "SDL_RENDER_METAL")
        set(SDL_SHARED OFF CACHE BOOL "SDL_SHARED")
        set(SDL_TEST OFF CACHE BOOL "SDL_TEST")
        set(SDL_TESTS OFF CACHE BOOL "SDL_TESTS")
        set(SDL_VIDEO OFF CACHE BOOL "SDL_VIDEO")
        set(SDL_VULKAN OFF CACHE BOOL "SDL_VULKAN")
        set(SDL_WASAPI OFF CACHE BOOL "SDL_WASAPI")

        set(SDL_HIGHDAPI_JOYSTICK ON CACHE BOOL "SDL_HIGHDAPI_JOYSTICK")
        set(SDL_JOYSTICK ON CACHE BOOL "SDL_JOYSTICK")
        set(SDL_STATIC ON CACHE BOOL "SDL_STATIC")
        set(SDL_XINPUT ON CACHE BOOL "SDL_XINPUT")

        set(FETCHCONTENT_QUIET OFF)
        FetchContent_MakeAvailable(SDL3-static)

        list(APPEND INCLUDES
            ${SDL3-static_SOURCE_DIR}/include
            )
        list(APPEND LIBS SDL3-static)

        set_target_properties(
            SDL3-static
            uninstall
            PROPERTIES FOLDER "SDL") 
    endif()
endif()

if (WOLF_SYSTEM_GAMEPAD_VIRTUAL)
  if (NOT WIN32)
    message(SEND_ERROR "WOLF_SYSTEM_GAMEPAD_VIRTUAL can only build for Windows")
  else()
    message("fetching https://github.com/ViGEm/ViGEmClient.git")
    FetchContent_Declare(
      ViGEmClient
      GIT_REPOSITORY https://github.com/ViGEm/ViGEmClient.git
      GIT_TAG        master
    )
    FetchContent_MakeAvailable(ViGEmClient)

    file(GLOB_RECURSE WOLF_SYSTEM_GAMEPAD_VIRTUAL_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_virtual_pool.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_virtual_pool.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_virtual.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/system/gamepad/w_gamepad_virtual.hpp"
    )
    list(APPEND SRCS ${WOLF_SYSTEM_GAMEPAD_VIRTUAL_SRCS})
    list(APPEND INCLUDES ${ViGEmClient_SOURCE_DIR}/include)
    list(APPEND LIBS 
      ViGEmClient
      Xinput.lib 
      SetupAPI.lib)
   endif()
endif()

if (WOLF_SYSTEM_LOG)
    if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_SYSTEM_LOG")
    endif()

    vcpkg_install(spdlog spdlog TRUE)
    list(APPEND LIBS spdlog::spdlog spdlog::spdlog_header_only)

    file(GLOB_RECURSE WOLF_SYSTEM_LOG_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/log/*"
    )
    list(APPEND SRCS ${WOLF_SYSTEM_LOG_SRC})
endif()

if (WOLF_SYSTEM_LZ4)
  if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_SYSTEM_LZ4")
  endif()
  vcpkg_install(lz4 lz4 TRUE)
  list(APPEND LIBS lz4::lz4)

  file(GLOB_RECURSE WOLF_SYSTEM_LZ4_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/system/compression/w_lz4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/system/compression/w_lz4.hpp"
  )
  list(APPEND SRCS ${WOLF_SYSTEM_LZ4_SRCS})
endif()

if (WOLF_SYSTEM_LZMA)
  if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_SYSTEM_LZMA")
  endif()
  message("fetching https://github.com/WolfEngine/lzma.git")
  FetchContent_Declare(
    lzma
    GIT_REPOSITORY https://github.com/WolfEngine/lzma.git
    GIT_TAG        main
  )
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_MakeAvailable(lzma)

  file(GLOB_RECURSE WOLF_SYSTEM_LZMA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/system/compression/w_lzma.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/system/compression/w_lzma.hpp"
  )
  list(APPEND SRCS ${WOLF_SYSTEM_LZMA_SRCS})
  list(APPEND INCLUDES ${lzma_SOURCE_DIR}/src)
  list(APPEND LIBS lzma)  
endif()

# include openSSL
if (WOLF_SYSTEM_OPENSSL AND NOT EMSCRIPTEN)
    vcpkg_install(OpenSSL openssl FALSE)
    list(APPEND LIBS OpenSSL::SSL OpenSSL::Crypto)
endif()

# include socket/websocket sources
if (WOLF_SYSTEM_SOCKET AND NOT EMSCRIPTEN)    
    file(GLOB_RECURSE WOLF_SYSTEM_SOCKET_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_socket_options.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_tcp_client.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_tcp_client.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_tcp_server.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_tcp_server.hpp"
    )
    list(APPEND SRCS ${WOLF_SYSTEM_SOCKET_SRC})
endif()

if (WOLF_SYSTEM_HTTP_WS)
    if (EMSCRIPTEN)
        file(GLOB_RECURSE WOLF_SYSTEM_HTTP_WS_SRC
            "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_ws_client_emc.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_ws_client_emc.hpp"
        )
    else()
        file(GLOB_RECURSE WOLF_SYSTEM_HTTP_WS_SRC
            "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_ws_client.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_ws_client.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_ws_server.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/system/socket/w_ws_server.hpp"
        )
    endif()
    list(APPEND SRCS ${WOLF_SYSTEM_HTTP_WS_SRC})
endif()

if (WOLF_SYSTEM_ZLIB)
    vcpkg_install(ZLIB zlib FALSE)
    list(APPEND LIBS ZLIB::ZLIB)
endif()

if (WOLF_SYSTEM_POSTGRESQL)
    vcpkg_install(libpq libpq TRUE)
    list(APPEND LIBS libpq::libpq)

    file(GLOB_RECURSE WOLF_SYSTEM_POSTGRESQL_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/db/w_postgresql.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/db/w_postgresql.hpp"
    )
    list(APPEND LIBS PostgreSQL::PostgreSQL)
endif()

if (WOLF_SYSTEM_LUA)
    vcpkg_install(Lua lua FALSE)
    vcpkg_install(sol2 sol2 TRUE)

    list(APPEND LIBS lua sol2)
    
    file(GLOB_RECURSE WOLF_SYSTEM_LUA_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/script/w_lua.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/script/w_lua.hpp"
    )
    
    list(APPEND SRCS 
        ${WOLF_SYSTEM_LUA_SRC}
    )
endif()

if (WOLF_SYSTEM_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Development)
    find_package(Boost REQUIRED COMPONENTS python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})

    file(GLOB_RECURSE WOLF_SYSTEM_PYTHON_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/script/w_python.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/script/w_python.hpp"
    )
    list(APPEND SRCS
        ${WOLF_SYSTEM_PYTHON_SRC}
    )
    list(APPEND INCLUDES ${Python3_INCLUDE_DIRS})
    list(APPEND LIBS Python3::Python Boost::python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})

    get_filename_component(PYTHON_HOME ${Python3_EXECUTABLE} DIRECTORY)
    add_definitions(
        -DBOOST_PYTHON_STATIC_LIB
        -DPYTHON_HOME="${PYTHON_HOME}"
    )
endif()

if (WOLF_SYSTEM_REDIS)
    if (ANDROID)
        message(WARNING "WOLF_SYSTEM_REDIS is not supported for Android")
    elseif(EMSCRIPTEN)
        message(WARNING "WOLF_SYSTEM_REDIS is not supported for the wasm32 target on Emscripten")
    else()
        if (NOT WOLF_SYSTEM_OPENSSL)
            message(FATAL "WOLF_SYSTEM_REDIS requires WOLF_SYSTEM_OPENSSL")
        endif()

        vcpkg_install_force(
            boost-json 
        )
        message("fetching https://github.com/boostorg/redis.git")
    
        FetchContent_Declare(
            boost_redis
            GIT_REPOSITORY https://github.com/boostorg/redis.git
            GIT_TAG        develop
        )
        
        set(FETCHCONTENT_QUIET OFF)
        FetchContent_MakeAvailable(boost_redis)

        list(APPEND INCLUDES ${boost_redis_SOURCE_DIR}/include) 
        
        file(GLOB_RECURSE WOLF_SYSTEM_REDIS_SRC
            "${CMAKE_CURRENT_SOURCE_DIR}/system/db/w_redis_client.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/system/db/w_redis_client.hpp"
        )

        list(APPEND SRCS 
            ${WOLF_SYSTEM_REDIS_SRC}
        )

        set_target_properties(
            boost_redis_src
            coverage
            cpp17_intro
            cpp17_intro_sync
            cpp20_containers
            cpp20_echo_server
            cpp20_intro
            cpp20_intro_tls
            cpp20_json
            cpp20_resolve_with_sentinel
            cpp20_streams
            cpp20_subscriber
            doc
            echo_server_client
            echo_server_direct
            test_conn_echo_stress
            test_conn_exec
            test_conn_exec_cancel
            test_conn_exec_cancel2
            test_conn_exec_error
            test_conn_exec_retry
            test_conn_push
            test_conn_quit
            test_conn_reconnect
            test_conn_run_cancel
            test_conn_tls
            test_issue_50
            test_low_level
            test_low_level_async
            test_request
            PROPERTIES FOLDER "boost-redis") 

    endif()
endif()

if (EMSCRIPTEN)
    file (GLOB_RECURSE WOLF_SYSTEM_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_gametime.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_gametime.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_trace.hpp"
    )
else()
    file (GLOB_RECURSE WOLF_SYSTEM_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/system/getopt.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_gametime.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_gametime.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_leak_detector.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_leak_detector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_process.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_process.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_time.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_time.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/w_trace.hpp"
    )
endif()

file(GLOB_RECURSE WOLF_SYSTEM_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/system/test/*"
)

list(APPEND SRCS 
    ${WOLF_SYSTEM_SRC} 
    ${WOLF_SYSTEM_TEST_SRC}
)
