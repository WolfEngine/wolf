/*
  Project: Wolf Engine. Copyright © 2014-2023 Pooya Eimandar
  https://github.com/WolfEngine/wolf
*/

#include <wolf/system/w_leak_detector.hpp>
#include <wolf/wolf.hpp>

#define BOOST_TEST_MODULE wolf
#include <boost/test/included/unit_test.hpp>
#include <cstdlib>
#include <filesystem>
#include <string_view>

constexpr const char* ENVKEY_CONTENT_PATH = "WOLF_CONTENT_PATH";
inline std::filesystem::path get_content_path(
    const std::string_view p_subpath) {
  auto env_content_path = wolf::get_env(ENVKEY_CONTENT_PATH);
  return env_content_path.empty()
             ? std::filesystem::current_path().append(p_subpath)
             : std::filesystem::path(env_content_path).append(p_subpath);
}

BOOST_AUTO_TEST_CASE(wolf_test) {
  const wolf::system::w_leak_detector _detector = {};

  std::cout << "entering test case 'wolf_test'" << std::endl;

  const auto _wolf_version = wolf::w_init();
  BOOST_TEST(_wolf_version.empty() == false);

  std::cout << "leaving test case 'wolf_test'" << std::endl;
}

#pragma region system tests

// #include <wolf/system/test/compress.hpp>
// #include <wolf/system/test/coroutine.hpp>
// #include <wolf/system/test/gamepad.hpp>
// #include <wolf/system/test/gametime.hpp>
// #include <wolf/system/test/log.hpp>
////#include <wolf/system/test/postgresql.hpp>
// #include <wolf/system/test/process.hpp>
// #include <wolf/system/test/signal_slot.hpp>
// #include <wolf/system/test/tcp.hpp>
// #include <wolf/system/test/trace.hpp>
// #include <wolf/system/test/ws.hpp>
// #include <wolf/system/test/lua.hpp>
// #include <wolf/system/test/python.hpp>
#include <wolf/system/test/redis.hpp>

#pragma endregion

#pragma region stream tests

// #include <wolf/stream/test/ffmpeg_stream.hpp>
// #include <wolf/stream/test/rist.hpp>
// #include <wolf/stream/test/grpc.hpp>
// #include <wolf/stream/test/quic.hpp>

#pragma endregion

#pragma region media tests

#include <wolf/media/test/avframe.hpp>
#include <wolf/media/test/ffmpeg.hpp>
// #include <wolf/media/test/image.hpp>
// #include <wolf/media/test/openal.hpp>
// #include <wolf/media/test/gstreamer.hpp>

#pragma endregion

#pragma region ml tests

// #include <wolf/ml/test/w_image_processor_test.hpp>
// #include <wolf/ml/test/w_ocr_engine_test.hpp>
// #include <wolf/ml/test/w_referee_test.hpp>
// #include <wolf/ml/test/w_soccer_test.hpp>
// #include <wolf/ml/test/w_utilities_test.hpp>
// #include <wolf/ml/test/w_nudity_detection_test.hpp>

#pragma endregion
