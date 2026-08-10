/* Minimal Prometheus exposition extractors for client tooling (dfkvctl and
 * dfkv_bench). Match exact metric names, with or without labels, and parse
 * integer-valued samples. Skips HELP/TYPE lines. Not on any datapath. */
#ifndef DFKV_PROM_PARSE_H_
#define DFKV_PROM_PARSE_H_

#include <cstdint>
#include <cstdlib>
#include <string>

namespace dfkv {

// Finds the first sample whose metric name exactly matches `name`. Returns
// false when absent, preserving the distinction between "missing" and zero.
inline bool PromMetricValue(const std::string& text, const std::string& name,
                            uint64_t* value) {
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    size_t len = (eol == std::string::npos) ? text.size() - pos : eol - pos;
    if (len > name.size() && text.compare(pos, name.size(), name) == 0) {
      char after = text[pos + name.size()];
      if (after == ' ' || after == '{') {
        size_t line_end = pos + len;
        size_t sp = text.rfind(' ', line_end - 1);
        if (sp != std::string::npos && sp + 1 < line_end) {
          if (value)
            *value = std::strtoull(text.c_str() + sp + 1, nullptr, 10);
          return true;
        }
      }
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return false;
}

// Back-compatible first-series accessor: missing remains zero.
inline uint64_t PromMetricValue(const std::string& text,
                                const std::string& name) {
  uint64_t value = 0;
  PromMetricValue(text, name, &value);
  return value;
}

// Sum every series in a family (for bounded labels such as {dev}).
inline uint64_t PromMetricSum(const std::string& text,
                              const std::string& name) {
  uint64_t total = 0;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    size_t len = (eol == std::string::npos) ? text.size() - pos : eol - pos;
    if (len > name.size() && text.compare(pos, name.size(), name) == 0) {
      char after = text[pos + name.size()];
      if (after == ' ' || after == '{') {
        size_t line_end = pos + len;
        size_t sp = text.rfind(' ', line_end - 1);
        if (sp != std::string::npos && sp + 1 < line_end)
          total += std::strtoull(text.c_str() + sp + 1, nullptr, 10);
      }
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return total;
}

}  // namespace dfkv

#endif  // DFKV_PROM_PARSE_H_
