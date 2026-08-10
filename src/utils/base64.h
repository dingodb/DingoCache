#ifndef DFKV_BASE64_H_
#define DFKV_BASE64_H_

#include <cstdint>
#include <string>

namespace dfkv {

// Standard base64 (RFC 4648) with '=' padding — etcd v3 JSON encodes all keys
// and values this way. Decode skips '=' and CR/LF; returns false on any other
// non-alphabet byte.
inline std::string Base64Encode(const std::string& in) {
  static const char* T =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((in.size() + 2) / 3 * 4);
  size_t i = 0;
  while (i + 3 <= in.size()) {
    uint32_t n = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8) | uint8_t(in[i + 2]);
    out += T[(n >> 18) & 63]; out += T[(n >> 12) & 63];
    out += T[(n >> 6) & 63];  out += T[n & 63];
    i += 3;
  }
  if (i + 1 == in.size()) {
    uint32_t n = uint32_t(uint8_t(in[i])) << 16;
    out += T[(n >> 18) & 63]; out += T[(n >> 12) & 63]; out += '='; out += '=';
  } else if (i + 2 == in.size()) {
    uint32_t n = (uint32_t(uint8_t(in[i])) << 16) | (uint8_t(in[i + 1]) << 8);
    out += T[(n >> 18) & 63]; out += T[(n >> 12) & 63]; out += T[(n >> 6) & 63]; out += '=';
  }
  return out;
}

inline bool Base64Decode(const std::string& in, std::string* out) {
  auto val = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  out->clear();
  uint32_t buf = 0;
  unsigned bits = 0;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r') continue;
    int v = val(c);
    if (v < 0) return false;
    buf = (buf << 6) | static_cast<uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out->push_back(char((buf >> bits) & 0xFF));
      buf &= bits == 0 ? 0u : (uint32_t{1} << bits) - 1;
    }
  }
  return true;
}

}  // namespace dfkv

#endif  // DFKV_BASE64_H_
