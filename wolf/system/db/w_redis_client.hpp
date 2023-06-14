#if defined(WOLF_SYSTEM_REDIS) && defined(WOLF_SYSTEM_SOCKET)

#pragma once

#include <hiredis/async.h>
#include <hiredis/hiredis.h>

#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/spawn.hpp>
#include <wolf/wolf.hpp>

namespace wolf::system::db {

struct w_redis_reply {
  ~w_redis_reply() {
    if (this->reply) {
      freeReplyObject(this->reply);
    }
  }
  redisReply* reply = nullptr;
};

class w_redis_client {
 public:
  W_API explicit w_redis_client(_In_ boost::asio::io_context& p_io_context)
      : _io_context(p_io_context) {}

  W_API ~w_redis_client() {
    if (this->_redis_context) {
      redisAsyncFree(this->_redis_context);
    }
  }

  W_API boost::asio::awaitable<boost::leaf::result<int>> connect(
      const std::string& p_host, std::uint16_t p_port) {
    if (p_host.empty()) {
      co_return W_FAILURE(std::errc::invalid_argument, "redis host is empty");
    }

    if (this->_redis_context) {
      redisAsyncFree(this->_redis_context);
    }

    this->_redis_context = redisAsyncConnect(p_host.c_str(), p_port);
    if (!this->_redis_context) {
      co_return W_FAILURE(std::errc::invalid_argument,
                          "failed to create hiredis context");
    }

    if (this->_redis_context->err) {
      co_return W_FAILURE(
          std::errc::invalid_argument,
          wolf::format("failed to create hiredis context because: {}",
                       this->_redis_context->errstr));
    }

    auto _ctx = this->_redis_context;
    boost::asio::co_spawn(
        this->_io_context,
        [&]() -> boost::asio::awaitable<void> {
          redisAsyncSetConnectCallback(
              _ctx, [](const redisAsyncContext*, int) { /*NOP*/ });
          redisAsyncSetDisconnectCallback(
              _ctx, [](const redisAsyncContext*, int) { /*NOP*/ });
          redisAsyncConnect(p_host.c_str(), p_port);

          co_return;
        },
        boost::asio::detached);

    co_return 0;
  }

  W_API boost::asio::awaitable<w_redis_reply> execute_command(
      _In_ const std::string& p_command) {
    w_redis_reply _reply;

    auto _ctx = this->_redis_context;
    boost::asio::co_spawn(
        this->_io_context,
        [&]() -> boost::asio::awaitable<void> {
          redisAsyncCommand(
              _ctx,
              [](redisAsyncContext* p_ctx, void* p_reply, void* p_arg) {
                auto _reply = static_cast<w_redis_reply*>(p_arg);
                if (_reply) {
                  auto _r = static_cast<redisReply*>(p_reply);
                  _reply->reply =
                      _r ? _r : reinterpret_cast<redisReply*>(REDIS_ERR);
                }
              },
              &_reply, p_command.c_str());

          co_return;
        },
        boost::asio::detached);

    co_return _reply;
  }

 private:
  boost::asio::io_context& _io_context;
  redisAsyncContext* _redis_context = nullptr;
};

}  // namespace wolf::system::db

#endif  // defined(WOLF_SYSTEM_REDIS) && defined(WOLF_SYSTEM_SOCKET)