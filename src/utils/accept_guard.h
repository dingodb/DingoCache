/* What a thread-per-connection accept loop does when it cannot take a
 * connection: accept() failed, the handler thread could not be created, or an
 * admission limit refused the peer.
 *
 * accept() fails in two different ways. Some errors belong to the one
 * connection that was at the head of the queue (the peer reset it before it
 * was accepted, a signal arrived): the loop simply accepts again. The others
 * say this process or host is out of descriptors or memory, and they do not go
 * away by retrying at once: the connection stays queued and accept() returns
 * the same error immediately, so a loop that just retries spins a core while
 * every other socket() and open() in the process fails too. OnAcceptError
 * counts those, logs them at a bounded rate and waits before the next attempt.
 *
 * std::thread throws when the system refuses a thread. StartThread turns that
 * into a return value so the loop can refuse one connection instead of the
 * exception terminating the process. */
#ifndef DFKV_ACCEPT_GUARD_H_
#define DFKV_ACCEPT_GUARD_H_

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "utils/log.h"
#include "utils/log_throttle.h"

namespace dfkv {
namespace net {

// Errors of the connection being accepted, not of this process. Linux hands
// network errors already pending on the new socket to accept(); accept(2) asks
// for them to be treated like EAGAIN.
inline bool AcceptErrorIsPerConnection(int err) {
  switch (err) {
    case EINTR:
    case EAGAIN:
#if EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK:
#endif
    case ECONNABORTED:
    case EPROTO:
    case EPERM:  // refused by a firewall rule
    case ENETDOWN:
    case ENOPROTOOPT:
    case EHOSTDOWN:
#ifdef ENONET
    case ENONET:
#endif
    case EHOSTUNREACH:
    case EOPNOTSUPP:
    case ENETUNREACH:
    case ETIMEDOUT:
      return true;
    default:
      return false;
  }
}

// Test seam: the next N StartThread calls fail as if the system had refused
// the thread. Zero in production.
inline std::atomic<int> g_fail_thread_starts_for_test{0};

// Starts `fn` on a new thread in *out. False when the thread could not be
// created; *out is then left as it was.
template <class Fn>
bool StartThread(std::thread* out, Fn&& fn) {
  try {
    int pending = g_fail_thread_starts_for_test.load(std::memory_order_relaxed);
    while (pending > 0 &&
           !g_fail_thread_starts_for_test.compare_exchange_weak(
               pending, pending - 1, std::memory_order_relaxed)) {
    }
    if (pending > 0)
      throw std::system_error(EAGAIN, std::generic_category());
    *out = std::thread(std::forward<Fn>(fn));
    return true;
  } catch (const std::system_error&) {
    return false;
  }
}

class AcceptGuard {
 public:
  // How long the loop stays away from accept() after a failure that is not
  // about one connection. Long enough to stop the spin, short enough that a
  // queued peer is picked up well inside its connect timeout once descriptors
  // are available again.
  static constexpr int kBackoffMs = 100;

  // `listener` names the accept loop in log lines.
  explicit AcceptGuard(const char* listener) : listener_(listener) {}

  // accept() returned -1 with `err`. Returns at once for an error of the one
  // connection. Otherwise counts, logs and waits up to kBackoffMs, returning
  // early when `running` clears so that stopping the listener is not delayed
  // by more than one poll step.
  void OnAcceptError(int err, const std::atomic<bool>& running) {
    if (AcceptErrorIsPerConnection(err)) return;
    accept_failures_.fetch_add(1, std::memory_order_relaxed);
    Log(kAcceptFailed, err,
        std::string("accept failed: ") + std::strerror(err) +
            "; retrying every " + std::to_string(kBackoffMs) + " ms");
    constexpr int kStepMs = 10;
    for (int waited = 0;
         waited < kBackoffMs && running.load(std::memory_order_relaxed);
         waited += kStepMs)
      std::this_thread::sleep_for(std::chrono::milliseconds(kStepMs));
  }

  // The handler thread for an accepted connection could not be created; the
  // caller closes that connection.
  void OnThreadStartFailure() {
    thread_start_failures_.fetch_add(1, std::memory_order_relaxed);
    Log(kThreadStartFailed, 0,
        "cannot create a connection thread; connection refused");
  }

  // An admission limit refused a connection; the caller counts and closes it.
  void LogRejected(const std::string& why) { Log(kRejected, 0, why); }

  uint64_t accept_failures() const {
    return accept_failures_.load(std::memory_order_relaxed);
  }
  uint64_t thread_start_failures() const {
    return thread_start_failures_.load(std::memory_order_relaxed);
  }

 private:
  enum Kind : uint64_t { kAcceptFailed = 1, kThreadStartFailed, kRejected };

  void Log(Kind kind, int err, const std::string& what) {
    const uint64_t now_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    uint64_t suppressed = 0;
    if (!log_.Admit(LogThrottle::Key(std::string(), kind,
                                     static_cast<uint64_t>(err)),
                    now_us, &suppressed))
      return;
    DFKV_LOG_WARN(std::string(listener_) + ": " + what + " (" +
                  std::to_string(suppressed) +
                  " earlier events not logged)");
  }

  const char* const listener_;
  std::atomic<uint64_t> accept_failures_{0};
  std::atomic<uint64_t> thread_start_failures_{0};
  LogThrottle log_;
};

}  // namespace net
}  // namespace dfkv

#endif  // DFKV_ACCEPT_GUARD_H_
