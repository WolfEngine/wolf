# fetch gRPC
if (WOLF_STREAM_GRPC)
    if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_STREAM_GRPC")
    endif()

    vcpkg_install(asio-grpc asio-grpc TRUE)
    list(APPEND LIBS asio-grpc::asio-grpc)

    file(GLOB_RECURSE WOLF_STREAM_GRPC_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/grpc/*"
    )

    list(APPEND SRCS 
        ${WOLF_STREAM_GRPC_SRC}
    )

    if(WOLF_TEST)
        add_library(generate-protos OBJECT)
        
        target_link_libraries(generate-protos PUBLIC protobuf::libprotobuf gRPC::grpc++_unsecure)
        
        set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/protos")
        set(PROTO_IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/protos")
        
        asio_grpc_protobuf_generate(
            GENERATE_GRPC GENERATE_MOCK_CODE
            TARGET generate-protos
            USAGE_REQUIREMENT PUBLIC
            IMPORT_DIRS ${PROTO_IMPORT_DIRS}
            OUT_DIR "${PROTO_BINARY_DIR}"
            PROTOS "${CMAKE_CURRENT_SOURCE_DIR}/protos/raft.proto")

        list(APPEND INCLUDES "${PROTO_BINARY_DIR}")
        list(APPEND TESTS_SRCS 
                "${PROTO_BINARY_DIR}/raft.grpc.pb.cc"
                "${PROTO_BINARY_DIR}/raft.pb.cc"
        )
    endif()

endif()

if (WOLF_STREAM_JANUS)

    file(GLOB_RECURSE WOLF_STREAM_JANUS_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/janus/*"
    )

    list(APPEND SRCS 
        ${WOLF_STREAM_JANUS_SRC}
    )

endif()

# fetch msquic
if (WOLF_STREAM_QUIC)
    if (EMSCRIPTEN)
        message(FATAL_ERROR "WOLF_STREAM_QUIC is not supported for wasm32 target")
    endif()

    if (NOT WIN32)
        message(FATAL_ERROR "WOLF_STREAM_QUIC feature is not avilable on non-windows yet.")
    endif()

    file(GLOB_RECURSE WOLF_STREAM_QUIC_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/stream/quic/*"
    )

    if (WIN32 OR WIN64)
        FetchContent_Declare(
            msquic
            URL https://github.com/microsoft/msquic/releases/download/v2.2.0/msquic_windows_x64_Release_schannel.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_Populate(msquic)
    else()
        message(FATAL_ERROR "WOLF_STREAM_QUIC feature is not supported on target platform.")
    endif()

    add_library(msquic-lib INTERFACE)
    add_library(msquic::msquic ALIAS msquic-lib)
    target_include_directories(msquic-lib INTERFACE ${msquic_SOURCE_DIR}/include)
    target_link_directories(msquic-lib INTERFACE BEFORE ${msquic_SOURCE_DIR}/bin)
    target_link_directories(msquic-lib INTERFACE BEFORE ${msquic_SOURCE_DIR}/lib)
    target_link_libraries(msquic-lib INTERFACE msquic)

    list(APPEND SRCS ${WOLF_STREAM_QUIC_SRCS})
    list(APPEND LIBS msquic::msquic)
endif()

if (WOLF_STREAM_RIST)
    if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_STREAM_RIST")
    endif()

    set(RIST_TARGET "rist")
    message("fetching https://code.videolan.org/rist/librist.git")
    FetchContent_Declare(
        ${RIST_TARGET}
        GIT_REPOSITORY https://code.videolan.org/rist/librist.git
        GIT_TAG        master
      )
    
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_MakeAvailable(${RIST_TARGET})
      
    if (ANDROID)
      add_custom_command(OUTPUT rist_command.out COMMAND
      /bin/bash "${CMAKE_CURRENT_SOURCE_DIR}/third_party/shells/librist/librist-android.sh" --build_dir=${rist_BINARY_DIR}
       WORKING_DIRECTORY ${rist_SOURCE_DIR})
      add_custom_target(rist ALL DEPENDS rist_command.out)
      
      list(APPEND LIBS
        ${rist_BINARY_DIR}/librist.a)
    else ()
      STRING(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
      
      add_custom_command(OUTPUT rist_command.out COMMAND cmd /c "meson setup ${rist_BINARY_DIR} --backend vs2022 --default-library static --buildtype ${CMAKE_BUILD_TYPE_LOWER} & meson compile -C ${rist_BINARY_DIR}" WORKING_DIRECTORY ${rist_SOURCE_DIR})
      add_custom_target(rist ALL DEPENDS rist_command.out)

      list(APPEND LIBS
        ws2_32
        ${rist_BINARY_DIR}/librist.a)
        
    endif()
    
    list(APPEND INCLUDES
        ${rist_BINARY_DIR}
        ${rist_BINARY_DIR}/include
        ${rist_BINARY_DIR}/include/librist
        ${rist_SOURCE_DIR}/contrib
        ${rist_SOURCE_DIR}/contrib/mbedtls/include
        ${rist_SOURCE_DIR}/include
        ${rist_SOURCE_DIR}/include/librist
        ${rist_SOURCE_DIR}/src
      )
      
      file(GLOB_RECURSE WOLF_STREAM_RIST_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/rist/*"
      )
      list(APPEND SRCS ${WOLF_STREAM_RIST_SRC})
endif()

if (WOLF_STREAM_WEBRTC)
    if (EMSCRIPTEN)
        message(FATAL_ERROR "the wasm32 target is not supported for WOLF_STREAM_WEBRTC")
    endif()

    # we need http & json for webrtc
    
    if (NOT WOLF_SYSTEM_JSON)
        message( FATAL_ERROR "'WOLF_STREAM_WEBRTC' needs 'WOLF_SYSTEM_JSON' = ON" )
    endif()

    if (NOT WOLF_STREAM_HTTP)
        message( FATAL_ERROR "'WOLF_STREAM_WEBRTC' needs 'WOLF_STREAM_HTTP' = ON" )
    endif()
   
    list(APPEND INCLUDES
        ${WEBRTC_SRC}
        ${WEBRTC_SRC}/third_party/abseil-cpp
        ${WEBRTC_SRC}/third_party/libyuv/include
    )
    if (WIN32)
        # enable/disable debug option
        if(CMAKE_BUILD_TYPE MATCHES Debug)
            add_definitions(
                -D_HAS_ITERATOR_DEBUGGING=1
                -D_ITERATOR_DEBUG_LEVEL=2
            )
        else()
            add_definitions(
                -D_HAS_ITERATOR_DEBUGGING=0
                -D_ITERATOR_DEBUG_LEVEL=0
            )
        endif()

        add_definitions(
            -DWEBRTC_WIN 
            -D__PRETTY_FUNCTION__=__FUNCTION__
            #-DUSE_X11 
            #-D_WINSOCKAPI_
            -DHAVE_SOUND) 

        list(APPEND LIBS
            d3d11 
            dmoguids 
            dwmapi
            dxgi 
            iphlpapi 
            msdmo 
            secur32 
            strmiids 
            winmm 
            wmcodecdspuuid 
        )
    elseif (APPLE)

        add_definitions(
            -DHAVE_SOUND 
            -DWEBRTC_MAC 
            -DWEBRTC_POSIX 
            -fno-rtti)
        
        find_library(APPLICATION_SERVICES ApplicationServices)
        find_library(AUDIO_TOOLBOX AudioToolBox)
        find_library(CORE_AUDIO CoreAudio)
        find_library(CORE_FOUNDATION CoreFoundation)
        find_library(CORE_SERVICES CoreServices)
        find_library(FOUNDATION Foundation)

        list(APPEND LIBS
            ${APPLICATION_SERVICES} 
            ${AUDIO_TOOLBOX} 
            ${CORE_AUDIO} 
            ${CORE_FOUNDATION} 
            ${CORE_SERVICES}
            ${FOUNDATION} 
        )
    endif()
    add_definitions(-DHAVE_JPEG)
    link_directories(${WEBRTC_SRC}/out/${TARGET_OS}/${CMAKE_BUILD_TYPE}/obj/)
    list(APPEND LIBS webrtc)

    file(GLOB_RECURSE WOLF_STREAM_WEBRTC_CAPTURER_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/webrtc/capturer/*"
    )
    file(GLOB_RECURSE WOLF_STREAM_WEBRTC_DATA_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/webrtc/data/*"
    )
    file(GLOB_RECURSE WOLF_STREAM_WEBRTC_INTERCEPTOR_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/webrtc/interceptor/*"
    )
    file(GLOB_RECURSE WOLF_STREAM_WEBRTC_MEDIA_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/webrtc/media/*"
    )
    file(GLOB_RECURSE WOLF_STREAM_WEBRTC_PEER_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream/webrtc/peer/*"
    )

    list(APPEND SRCS 
        ${WOLF_STREAM_WEBRTC_CAPTURER_SRC}
        ${WOLF_STREAM_WEBRTC_DATA_SRC}
        ${WOLF_STREAM_WEBRTC_INTERCEPTOR_SRC}
        ${WOLF_STREAM_WEBRTC_MEDIA_SRC}
        ${WOLF_STREAM_WEBRTC_PEER_SRC}
    )
endif()

file(GLOB_RECURSE WOLF_STREAM_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/stream/test/*"
)
list(APPEND SRCS ${WOLF_STREAM_TEST_SRC})

