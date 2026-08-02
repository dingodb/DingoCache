#include "common/namespace.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>


namespace dfkv {
namespace {

constexpr size_t kMaxFieldBytes = 1u << 20;
constexpr size_t kMaxComponentBytes = 256;

void SetError(std::string* error, const std::string& value) {
  if (error) *error = value;
}

bool ValidRequiredField(const std::string& value, const char* name,
                        std::string* error) {
  if (value.empty()) {
    SetError(error, std::string(name) + " must not be empty");
    return false;
  }
  if (value.size() > kMaxFieldBytes) {
    SetError(error, std::string(name) + " exceeds 1 MiB");
    return false;
  }
  return true;
}

void AppendU16(std::string* out, uint16_t value) {
  char bytes[2];
  bytes[0] = static_cast<char>(value);
  bytes[1] = static_cast<char>(value >> 8);
  out->append(bytes, sizeof(bytes));
}

void AppendU32(std::string* out, uint32_t value) {
  char bytes[4];
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    bytes[i] = static_cast<char>(value >> (8 * i));
  }
  out->append(bytes, sizeof(bytes));
}

void AppendU64(std::string* out, uint64_t value) {
  char bytes[8];
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    bytes[i] = static_cast<char>(value >> (8 * i));
  }
  out->append(bytes, sizeof(bytes));
}

void AppendString(std::string* out, const std::string& value) {
  AppendU32(out, static_cast<uint32_t>(value.size()));
  out->append(value);
}

void AppendRank(std::string* out, const RankCoordinate& coordinate) {
  AppendU32(out, coordinate.world_size);
  AppendU32(out, static_cast<uint32_t>(coordinate.rank));
}

}  // namespace

Status RankCoordinate::Validate(const char* axis, std::string* error) const {
  if (world_size == 0) {
    SetError(error, std::string(axis) + " world_size must be positive");
    return Status::kInvalid;
  }
  if (rank < -1 || (rank >= 0 && static_cast<uint32_t>(rank) >= world_size)) {
    SetError(error, std::string(axis) + " rank must be -1 or within world_size");
    return Status::kInvalid;
  }
  return Status::kOk;
}

Status NamespaceDescriptor::Validate(std::string* error) const {
  if (!ValidRequiredField(tenant_id, "tenant_id", error) ||
      !ValidRequiredField(model_id, "model_id", error) ||
      !ValidRequiredField(model_revision, "model_revision", error) ||
      !ValidRequiredField(pool_name, "pool_name", error) ||
      !ValidRequiredField(dtype, "dtype", error)) {
    return Status::kInvalid;
  }
  if (layout_fingerprint == 0) {
    SetError(error, "layout_fingerprint must be nonzero");
    return Status::kInvalid;
  }
  if (block_tokens == 0) {
    SetError(error, "block_tokens must be positive");
    return Status::kInvalid;
  }
  if (layer_count == 0 || layer_begin >
                              std::numeric_limits<uint32_t>::max() - layer_count) {
    SetError(error, "layer range must be nonempty and must not overflow");
    return Status::kInvalid;
  }
  if (tp.Validate("tp", error) != Status::kOk ||
      dp.Validate("dp", error) != Status::kOk ||
      pp.Validate("pp", error) != Status::kOk) {
    return Status::kInvalid;
  }
  return Status::kOk;
}

std::string NamespaceDescriptor::Serialize() const {
  static constexpr char kMagic[8] = {'D', 'F', 'K', 'V', 'N', 'S', '\0', '\2'};
  std::string out;
  out.reserve(sizeof(kMagic) + 5 * 4 + tenant_id.size() + model_id.size() +
              model_revision.size() + pool_name.size() + dtype.size() + 62);
  out.append(kMagic, sizeof(kMagic));
  AppendString(&out, tenant_id);
  AppendString(&out, model_id);
  AppendString(&out, model_revision);
  AppendU16(&out, static_cast<uint16_t>(pool_kind));
  AppendString(&out, pool_name);
  AppendString(&out, dtype);
  AppendU64(&out, layout_fingerprint);
  AppendU32(&out, block_tokens);
  AppendU32(&out, group_id);
  AppendU32(&out, layer_begin);
  AppendU32(&out, layer_count);
  AppendRank(&out, tp);
  AppendRank(&out, dp);
  AppendRank(&out, pp);
  return out;
}

Status ObjectDescriptor::Validate(std::string* error) const {
  if (!ValidRequiredField(logical_key, "logical_key", error) ||
      !ValidRequiredField(component, "component", error)) {
    return Status::kInvalid;
  }
  if (component.size() > kMaxComponentBytes) {
    SetError(error, "component exceeds 256 bytes");
    return Status::kInvalid;
  }
  if (chunk_count == 0 || chunk_index >= chunk_count) {
    SetError(error, "chunk_index must be within nonzero chunk_count");
    return Status::kInvalid;
  }
  return Status::kOk;
}

std::string ObjectDescriptor::Serialize(const NamespaceDescriptor& ns) const {
  static constexpr char kMagic[8] = {'D', 'F', 'K', 'V', 'O', 'B', 'J', '2'};
  const std::string namespace_bytes = ns.Serialize();
  std::string out;
  out.reserve(sizeof(kMagic) + 4 + namespace_bytes.size() + 8 +
              logical_key.size() + component.size());
  out.append(kMagic, sizeof(kMagic));
  AppendString(&out, namespace_bytes);
  AppendString(&out, logical_key);
  AppendString(&out, component);
  AppendU32(&out, chunk_index);
  AppendU32(&out, chunk_count);
  return out;
}

}  // namespace dfkv
