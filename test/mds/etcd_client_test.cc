#include "mds/etcd_client.h"
#include "utils/base64.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
using namespace dfkv;  // NOLINT

namespace {
struct FakeHttp : HttpTransport {
  struct Call { std::string path, body; };
  std::vector<Call> calls;
  std::vector<HttpResponse> canned;
  size_t idx = 0;
  bool Post(const std::string& path, const std::string& body, HttpResponse* out) override {
    calls.push_back({path, body});
    if (idx >= canned.size()) return false;
    *out = canned[idx++];
    return true;
  }
};
}  // namespace

TEST(EtcdClient, LeaseGrantParsesId) {
  FakeHttp h;
  h.canned.push_back({200, R"({"ID":"7587847878","TTL":"30"})"});
  EtcdClient c(&h);
  auto id = c.LeaseGrant(30);
  ASSERT_TRUE(id.has_value());
  EXPECT_EQ(*id, 7587847878LL);
  EXPECT_EQ(h.calls[0].path, "/v3/lease/grant");
  EXPECT_NE(h.calls[0].body.find("\"TTL\""), std::string::npos);
  EXPECT_NE(h.calls[0].body.find("30"), std::string::npos);
}

TEST(EtcdClient, PutBase64sKeyAndValueWithLease) {
  FakeHttp h;
  h.canned.push_back({200, R"({"header":{"revision":"9"}})"});
  EtcdClient c(&h);
  ASSERT_TRUE(c.Put("/dfkv/v1/groups/g/members/n1", "payload", 42));
  EXPECT_EQ(h.calls[0].path, "/v3/kv/put");
  EXPECT_NE(h.calls[0].body.find(Base64Encode("/dfkv/v1/groups/g/members/n1")), std::string::npos);
  EXPECT_NE(h.calls[0].body.find(Base64Encode("payload")), std::string::npos);
  EXPECT_NE(h.calls[0].body.find("\"lease\":\"42\""), std::string::npos);
}

TEST(EtcdClient, RangePrefixDecodesKvsAndRevision) {
  FakeHttp h;
  std::string body = std::string("{\"header\":{\"revision\":\"12\"},\"kvs\":[") +
      "{\"key\":\"" + Base64Encode("/p/n1") + "\",\"value\":\"" + Base64Encode("v1") + "\"}," +
      "{\"key\":\"" + Base64Encode("/p/n2") + "\",\"value\":\"" + Base64Encode("v2") + "\"}]}";
  h.canned.push_back({200, body});
  EtcdClient c(&h);
  auto r = c.RangePrefix("/p/");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->revision, 12);
  ASSERT_EQ(r->kvs.size(), 2u);
  EXPECT_EQ(r->kvs[0].first, "/p/n1");
  EXPECT_EQ(r->kvs[0].second, "v1");
  EXPECT_EQ(r->kvs[1].second, "v2");
  EXPECT_NE(h.calls[0].body.find(Base64Encode("/p0")), std::string::npos);
}

TEST(EtcdClient, RangePrefixEmptyResultOkWithRevision) {
  FakeHttp h;
  h.canned.push_back({200, R"({"header":{"revision":"3"}})"});
  EtcdClient c(&h);
  auto r = c.RangePrefix("/none/");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->revision, 3);
  EXPECT_TRUE(r->kvs.empty());
}

TEST(EtcdClient, Non200OrTransportFailureReturnsNullopt) {
  FakeHttp h;
  h.canned.push_back({500, "boom"});
  EtcdClient c(&h);
  EXPECT_FALSE(c.LeaseGrant(30).has_value());
  EXPECT_FALSE(c.LeaseGrant(30).has_value());
}

TEST(EtcdClient, MetricsCoverEveryOperationAndErrors) {
  FakeHttp h;
  h.canned = {
      {200, R"({"ID":"1","TTL":"30"})"},
      {200, R"({"result":{"TTL":"30"}})"},
      {200, "{}"},
      {503, "unavailable"},
      {200, R"({"header":{"revision":"4"}})"},
  };
  EtcdClient c(&h);
  EXPECT_TRUE(c.LeaseGrant(30).has_value());
  EXPECT_TRUE(c.LeaseKeepAlive(1));
  EXPECT_TRUE(c.Put("/k", "v", 1));
  EXPECT_FALSE(c.RangePrefix("/").has_value());
  EXPECT_TRUE(c.RangePrefixForMetrics("/dfkv/v1/groups/").has_value());
  const std::string text = c.MetricsText();
  EXPECT_NE(text.find("dfkv_mds_etcd_requests_total{op=\"lease_grant\"} 1"),
            std::string::npos) << text;
  EXPECT_NE(text.find("dfkv_mds_etcd_requests_total{op=\"lease_keepalive\"} 1"),
            std::string::npos) << text;
  EXPECT_NE(text.find("dfkv_mds_etcd_requests_total{op=\"put\"} 1"),
            std::string::npos) << text;
  EXPECT_NE(text.find("dfkv_mds_etcd_requests_total{op=\"range\"} 1"),
            std::string::npos) << text;
  EXPECT_NE(text.find("dfkv_mds_etcd_request_errors_total{op=\"range\"} 1"),
            std::string::npos) << text;
  EXPECT_NE(text.find(
                "dfkv_mds_etcd_request_duration_seconds_count{op=\"range\"} 1"),
            std::string::npos) << text;
  EXPECT_NE(text.find(
                "dfkv_mds_etcd_requests_total{op=\"metrics_range\"} 1"),
            std::string::npos) << text;
}

TEST(EtcdClient, PrefixRangeEndAll0xFFScansToEnd) {
  // All-0xFF prefix => range_end must be "\0" (scan to keyspace end), NOT empty
  // (empty range_end == single-key get in etcd).
  FakeHttp h;
  h.canned.push_back({200, R"({"header":{"revision":"1"}})"});
  EtcdClient c(&h);
  ASSERT_TRUE(c.RangePrefix(std::string("\xff\xff", 2)).has_value());
  EXPECT_NE(h.calls[0].body.find(Base64Encode(std::string("\0", 1))), std::string::npos);
}

TEST(EtcdClient, IntegrationRealEtcd) {
  const char* ep = std::getenv("DFKV_TEST_ETCD");
  if (!ep) GTEST_SKIP() << "set DFKV_TEST_ETCD=host:port to run";
  TcpHttpTransport t(ep, 2000);
  EtcdClient c(&t);
  auto id = c.LeaseGrant(30);
  ASSERT_TRUE(id.has_value());
  std::string key = "/dfkv/itest/n1";
  ASSERT_TRUE(c.Put(key, "hello", *id));
  ASSERT_TRUE(c.LeaseKeepAlive(*id));
  auto r = c.RangePrefix("/dfkv/itest/");
  ASSERT_TRUE(r.has_value());
  bool found = false;
  for (auto& kv : r->kvs) if (kv.first == key && kv.second == "hello") found = true;
  EXPECT_TRUE(found);
  EXPECT_GT(r->revision, 0);
}
