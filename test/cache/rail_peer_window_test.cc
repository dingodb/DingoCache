#include "cache/rail_peer_window.h"

#include <gtest/gtest.h>

#include <string>

namespace dfkv {
namespace {

TEST(PeerWindow, CountsDistinctHostsOnce) {
  PeerWindow window(8, 30000);
  window.Note("10.0.0.1", 1000);
  window.Note("10.0.0.1", 2000);  // the same host again
  window.Note("10.0.0.2", 3000);
  EXPECT_EQ(window.Count(3000), 2u);
}

TEST(PeerWindow, HostsLeaveTheWindow) {
  PeerWindow window(8, 30000);
  window.Note("10.0.0.1", 1000);
  window.Note("10.0.0.2", 20000);
  EXPECT_EQ(window.Count(31000), 2u);   // the first is exactly 30 s old
  EXPECT_EQ(window.Count(31001), 1u);   // and then out
  EXPECT_EQ(window.Count(51001), 0u);
  window.Note("10.0.0.1", 60000);  // back
  EXPECT_EQ(window.Count(60000), 1u);
}

TEST(PeerWindow, BoundedByCapacityKeepingTheMostRecent) {
  PeerWindow window(3, 30000);
  for (int i = 0; i < 10; ++i)
    window.Note("10.0.0." + std::to_string(i), 1000 + i);
  EXPECT_EQ(window.Count(1010), 3u);
}

TEST(PeerWindow, NothingNotedForAnEmptyHost) {
  PeerWindow window(3, 30000);
  window.Note("", 1000);
  EXPECT_EQ(window.Count(1000), 0u);
}

TEST(PeerHost, StripsThePort) {
  EXPECT_EQ(PeerHost("192.0.2.7:4800"), "192.0.2.7");
  EXPECT_EQ(PeerHost("[2001:db8::1]:4800"), "2001:db8::1");
  EXPECT_EQ(PeerHost("unknown"), "");
  EXPECT_EQ(PeerHost(""), "");
}

}  // namespace
}  // namespace dfkv
