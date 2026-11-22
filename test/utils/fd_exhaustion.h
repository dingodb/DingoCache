/* Test helpers for "the process is out of file descriptors".
 *
 * The condition is produced for real, by lowering RLIMIT_NOFILE to the number
 * of descriptors already open, and only inside a forked child: the limit is a
 * process property, so the test binary and the shell that started it keep
 * theirs, and nothing is actually opened to get there. */
#ifndef DFKV_TEST_FD_EXHAUSTION_H_
#define DFKV_TEST_FD_EXHAUSTION_H_

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <ctime>
#include <functional>

namespace dfkv {
namespace testing_fd {

// Runs `body` in a forked child and returns what it returned (0..255), or -1
// when the child did not exit normally. A child still running after
// `timeout_s` is killed and reported as -1, so one that deadlocks fails its
// test instead of holding the run until the harness gives up. The child leaves
// through _exit, so it runs none of the parent's atexit handlers or test
// teardown.
inline int RunInChild(const std::function<int()>& body, int timeout_s = 60) {
  const pid_t child = ::fork();
  if (child < 0) return -1;
  if (child == 0) ::_exit(body());
  int status = 0;
  const timespec step{0, 10 * 1000 * 1000};
  for (int waited_ms = 0;; waited_ms += 10) {
    const pid_t waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child) break;
    if (waited < 0 && errno != EINTR) return -1;
    if (waited_ms >= timeout_s * 1000) {
      ::kill(child, SIGKILL);
      while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
      }
      return -1;
    }
    ::nanosleep(&step, nullptr);
  }
  if (!WIFEXITED(status)) return -1;
  return WEXITSTATUS(status);
}

// The number the next new descriptor of this process would get, or -1. A test
// that needs another thread's descriptor to be closed before it goes on waits
// for this to return to the value it had before that descriptor was opened.
// The probe itself holds the number only for the duration of the call.
inline int LowestFreeDescriptor() {
  // open() returns the lowest free number, so nothing below it is free.
  const int lowest_free = ::open("/dev/null", O_RDONLY);
  if (lowest_free >= 0) ::close(lowest_free);
  return lowest_free;
}

// While alive, every attempt to get a new descriptor in this process fails
// with EMFILE; descriptors that are already open keep working. Create the
// sockets a test still needs before constructing this.
class NoFreeDescriptors {
 public:
  NoFreeDescriptors() {
    const int lowest_free = LowestFreeDescriptor();
    if (lowest_free < 0) return;
    if (::getrlimit(RLIMIT_NOFILE, &saved_) != 0) return;
    rlimit lowered = saved_;
    lowered.rlim_cur = static_cast<rlim_t>(lowest_free);
    active_ = ::setrlimit(RLIMIT_NOFILE, &lowered) == 0;
  }
  ~NoFreeDescriptors() { Restore(); }
  bool active() const { return active_; }
  void Restore() {
    if (active_) ::setrlimit(RLIMIT_NOFILE, &saved_);
    active_ = false;
  }

 private:
  rlimit saved_{};
  bool active_ = false;
};

// CPU time this process has used, all threads.
inline double ProcessCpuSeconds() {
  timespec ts{};
  ::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + ts.tv_nsec / 1e9;
}

}  // namespace testing_fd
}  // namespace dfkv

#endif  // DFKV_TEST_FD_EXHAUSTION_H_
