#if defined(WOLF_SYSTEM_REDIS) && defined(WOLF_SYSTEM_SOCKET)

#pragma once

#include <boost/asio.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>
#include <boost/redis.hpp>
#include <boost/redis/connection.hpp>
#include <boost/redis/src.hpp>
#include <wolf/wolf.hpp>

namespace wolf::system::db {

using namespace boost::asio::experimental::awaitable_operators;

class w_redis_client {
  using connection =
      boost::asio::deferred_t::as_default_on_t<boost::redis::connection>;

 public:
  W_API explicit w_redis_client(_In_ boost::redis::config p_config)
      : _config(std::move(p_config)) {}
  W_API ~w_redis_client() { cancel(); }

  // disable copy constructor
  w_redis_client(const w_redis_client&) = delete;
  // disable copy operator
  w_redis_client& operator=(const w_redis_client&) = delete;

  // move constructor
  w_redis_client(w_redis_client&& p_other) {
    _move(std::forward<w_redis_client&&>(p_other));
  }
  // move operator
  w_redis_client& operator=(w_redis_client&& p_other) noexcept {
    _move(std::forward<w_redis_client&&>(p_other));
    return *this;
  }

  W_API auto connect() -> boost::asio::awaitable<boost::leaf::result<int>> {
    try {
      if (!this->_conn) {
        this->_conn = std::make_shared<connection>(
            co_await boost::asio::this_coro::executor);
        if (!this->_conn) {
          co_return W_FAILURE(std::errc::not_enough_memory,
                              "could not allocate memory for redis connection");
        }
      }

      this->_conn->async_run(
          this->_config, {},
          boost::asio::consign(boost::asio::detached, this->_conn));

      boost::redis::response<std::string> _res;
      boost::redis::request _req;
      _req.push("PING");

      std::ignore = co_await this->_conn->async_exec(
          _req, _res, boost::asio::use_awaitable);

      if (std::get<0>(_res).value() != "PONG") {
        const auto _msg = wolf::format(
            "could not connect to redis host '{}:{}' because 'PONG' response "
            "was "
            "expected",
            this->_config.addr.host, this->_config.addr.port);
        co_return W_FAILURE(std::errc::operation_canceled, _msg);
      }

      co_return 0;
    } catch (const std::exception& e) {
      const auto _msg = wolf::format(
          "could not connect to redis host '{}:{}' because of {} ",
          this->_config.addr.host, this->_config.addr.port, e.what());
      co_return W_FAILURE(std::errc::operation_canceled, _msg);
    }
  }

  W_API void cancel() {
    if (this->_conn) {
      this->_conn->cancel();
    }
  }

  template <class R>
  auto exec(const boost::redis::request& p_req, _Inout_ R& p_res)
      -> boost::asio::awaitable<boost::leaf::result<int>> {
    try {
      co_return co_await this->_conn->async_exec(p_req, p_res,
                                                 boost::asio::use_awaitable);
    } catch (const std::exception& e) {
      const auto _msg = wolf::format(
          "could not execute redis command on host '{}:{}' because of {} ",
          this->_config.addr.host, this->_config.addr.port, e.what());
      co_return W_FAILURE(std::errc::operation_canceled, _msg);
    }
  }

 private:
  void _move(w_redis_client&& p_other) noexcept {
    if (this == &p_other) {
      return;
    }
    this->_config = std::move(_config);
    this->_conn = std::move(p_other._conn);
  }

  boost::redis::config _config;
  std::shared_ptr<connection> _conn = nullptr;
};

}  // namespace wolf::system::db

#endif  // defined(WOLF_SYSTEM_REDIS) && defined(WOLF_SYSTEM_SOCKET)