if(WOLF_ML_OCR OR WOLF_ML_NUDITY_DETECTION)
    file(GLOB_RECURSE WOLF_ML_SHARED_SRC
        # "${CMAKE_CURRENT_SOURCE_DIR}/ml/w_common.*"
        "${CMAKE_CURRENT_SOURCE_DIR}/ml/w_utilities.*"
        "${CMAKE_CURRENT_SOURCE_DIR}/ml/w_read_video_frames.*"
    )

    list(APPEND SRCS
        ${WOLF_ML_SHARED_SRC}
    )

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(DEBUG_LIB_EXTENTION "d")
    else()
        set(DEBUG_LIB_EXTENTION "")
    endif()

    # fetch opencv
    message("fetching https://github.com/opencv/opencv.git")
    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.5.4
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )   
    if(WIN64)
        FetchContent_GetProperties(opencv)  
        set(BUILD_LIST core,highgui,videoio CACHE STRING "BUILD_LIST")
        set(WITH_IPP OFF CACHE BOOL "WITH_IPP")
        set(BUILD_EXAMPLES OFF CACHE BOOL "BUILD_EXAMPLES")
        set(OPENCV_GENERATE_PKGCONFIG ON CACHE BOOL "OPENCV_GENERATE_PKGCONFIG")    
        FetchContent_MakeAvailable(opencv)  
        list(APPEND INCLUDES
            ${CMAKE_BINARY_DIR}
            ${opencv_SOURCE_DIR}/include
            ${opencv_SOURCE_DIR}/modules/core/include
            ${opencv_SOURCE_DIR}/modules/highgui/include
            ${opencv_SOURCE_DIR}/modules/imgcodecs/include
            ${opencv_SOURCE_DIR}/modules/imgproc/include
            ${opencv_SOURCE_DIR}/modules/videoio/include
        )
        list(APPEND LIBS
            ${opencv_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/opencv_core454${DEBUG_LIB_EXTENTION}.lib
            ${opencv_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/opencv_highgui454${DEBUG_LIB_EXTENTION}.lib
            ${opencv_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/opencv_imgcodecs454${DEBUG_LIB_EXTENTION}.lib
            ${opencv_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/opencv_imgproc454${DEBUG_LIB_EXTENTION}.lib
            ${opencv_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/opencv_videoio454${DEBUG_LIB_EXTENTION}.lib
        )
    elseif(LINUX)
        FetchContent_Populate(opencv)   
        list(APPEND INCLUDES
            ${opencv_BINARY_DIR}/install/include/opencv4
        )
        list(APPEND LIBS
            ${opencv_BINARY_DIR}/install/lib/libopencv_core.so
            ${opencv_BINARY_DIR}/install/lib/libopencv_highgui.so
            ${opencv_BINARY_DIR}/install/lib/libopencv_imgcodecs.so
            ${opencv_BINARY_DIR}/install/lib/libopencv_imgproc.so
            ${opencv_BINARY_DIR}/install/lib/libopencv_videoio.so
        )   
        add_custom_command(OUTPUT opencv_config.out COMMAND cmake -B ${opencv_BINARY_DIR} -DBUILD_LIST=core,highgui,videoio -DBUILD_opencv_python3=OFF -DWITH_IPP=OFF -DBUILD_EXAMPLES=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_INSTALL_PREFIX:PATH=${opencv_BINARY_DIR}/install ${opencv_SOURCE_DIR})
        add_custom_target(opencv_config ALL DEPENDS opencv_config.out)
        add_custom_command(OUTPUT opencv_build.out COMMAND cmake --build ${opencv_BINARY_DIR} --target install)
        add_custom_target(opencv_build ALL DEPENDS opencv_build.out)
    endif()

endif()

if(WOLF_ML_OCR)

    # install tesseract
    vcpkg_install(tesseract tesseract TRUE)
    list(APPEND LIBS Tesseract::libtesseract)

    # install tesseract
    vcpkg_install(rapidjson rapidjson TRUE)
    list(APPEND LIBS rapidjson)

    file(GLOB_RECURSE WOLF_ML_OCR_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/ml/referee_ocr/*"
    )

    list(APPEND SRCS
        ${WOLF_ML_OCR_SRC}
    )
endif()

if(WOLF_ML_NUDITY_DETECTION)
    # Set the C++ standard for the rest of the project
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Find Torch package
    find_package(Torch REQUIRED)

    # Set the C++ standard for the rest of the project
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # find_package(OpenCV REQUIRED)

    # list(APPEND INCLUDES
    #     ${OpenCV_INCLUDE_DIRS}
    # )

    list(APPEND LIBS
        ${TORCH_LIBRARIES}
        # ${OpenCV_LIBS}
    )

    # file(GLOB_RECURSE WOLF_ML_SHARED_SRC
    #     "${CMAKE_CURRENT_SOURCE_DIR}/ml/w_utilities.*"
    # )

    # list(APPEND SRCS
    #     ${WOLF_ML_SHARED_SRC}
    # )

    file(GLOB_RECURSE WOLF_ML_NUD_DET_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/ml/nudity_detection/*"
    )

    list(APPEND SRCS
        ${WOLF_ML_NUD_DET_SRC}
    )
endif()