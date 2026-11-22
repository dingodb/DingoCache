// CRC-32C (Castagnoli, reflected polynomial 0x82F63B78, initial value and
// final xor 0xFFFFFFFF), the checksum of iSCSI, ext4 metadata and btrfs.
// Crc32c("123456789") is 0xE3069283.
//
// Computed with the CPU's CRC32C instruction where there is one (SSE4.2 on
// x86-64, the CRC extension on AArch64), chosen once at run time, and by an
// eight-table software loop otherwise. Both give the same value for every
// input; Crc32cSoftware is exposed so that tests can compare them.
#pragma once

#include <cstddef>
#include <cstdint>

namespace dfkv {

// Continue a CRC over more bytes: Crc32c(b, Crc32c(a)) == Crc32c(a ++ b).
// The `crc` argument is a previous result (0 to start).
uint32_t Crc32c(const void* data, size_t len, uint32_t crc = 0);
uint32_t Crc32cSoftware(const void* data, size_t len, uint32_t crc = 0);
// Whether Crc32c uses the CPU instruction on this machine.
bool Crc32cHardware();

}  // namespace dfkv
