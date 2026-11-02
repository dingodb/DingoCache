# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire contract for the LookupKey ZMQ admin channel.

This is the single source of truth shared by ``LookupKeyClient`` and
``LookupKeyServer`` on the scheduler<->worker rank-0 admin channel.

Wire format (REQ/REP over IPC):

    Request: [msg_type: bytes] [payload_frames...]

      msg_type == LOOKUP_MSG:
          frame 1: token_len (u32 big-endian, 4 bytes)
          frame 2: hash_len (u16 big-endian, 2 bytes)
          frame 3: the block hashes concatenated without separators
          frame 4: retain_ttl_s (u32 big-endian; 0 = best-effort request)
        Response: [hit_count: u32 big-endian][applied_ttl_s: u32 big-endian]
          applied_ttl_s is the retention the worker will apply to this
          request's writes (0 = none: no TTL, unsupported ring, or precheck
          downgrade).

      msg_type == RESET_MSG:
          (no payload frames)
        Response: [RESP_OK] or [RESP_ERR]

The first frame of every request is a named bytes tag (not a numeric
sentinel that aliases the data field) so the protocol stays
self-describing and extensible. Lookup frames deliberately carry hashes as
one binary blob: the receiver derives the count from ``len(blob) / hash_len``.
"""

from collections.abc import Sequence


# Request message-type tags. Frame 0 of every request.
LOOKUP_MSG: bytes = b"lookup"
RESET_MSG: bytes = b"reset"

# Fixed field widths in the lookup request and response.
TOKEN_LEN_BYTES = 4
HASH_LEN_BYTES = 2
RETAIN_TTL_BYTES = 4
LOOKUP_RESPONSE_BYTES = 8

# Single-byte response status codes for admin commands.
RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"


def _encode_lookup_request(
    token_len: int,
    block_hashes: Sequence[bytes],
    retain_ttl_s: int = 0,
) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    """Encode a lookup request as five ZMQ frames.

    Empty hash lists use ``hash_len == 0`` and an empty blob. Non-empty hashes
    must have one homogeneous, non-zero width so the receiver can split the
    blob without per-hash framing.
    """
    if not 0 <= token_len < 1 << (8 * TOKEN_LEN_BYTES):
        raise ValueError(f"token_len is outside uint32: {token_len}")

    hash_len = len(block_hashes[0]) if block_hashes else 0
    if hash_len >= 1 << (8 * HASH_LEN_BYTES):
        raise ValueError(f"block hash is too long for uint16: {hash_len}")
    if block_hashes and hash_len == 0:
        raise ValueError("non-empty block hash list contains an empty hash")
    if any(len(block_hash) != hash_len for block_hash in block_hashes):
        raise ValueError("block hashes must all have the same length")
    if not 0 <= retain_ttl_s < 1 << (8 * RETAIN_TTL_BYTES):
        raise ValueError(f"retain_ttl_s is outside uint32: {retain_ttl_s}")

    return (
        LOOKUP_MSG,
        token_len.to_bytes(TOKEN_LEN_BYTES, byteorder="big"),
        hash_len.to_bytes(HASH_LEN_BYTES, byteorder="big"),
        b"".join(block_hashes),
        retain_ttl_s.to_bytes(RETAIN_TTL_BYTES, byteorder="big"),
    )


def _encode_lookup_response(hit_count: int, applied_ttl_s: int) -> bytes:
    return hit_count.to_bytes(4, byteorder="big") + applied_ttl_s.to_bytes(
        4, byteorder="big")


def _decode_lookup_response(resp: bytes) -> tuple[int, int]:
    if len(resp) != LOOKUP_RESPONSE_BYTES:
        raise ValueError(
            f"lookup response has {len(resp)} bytes, "
            f"expected {LOOKUP_RESPONSE_BYTES}"
        )
    return (int.from_bytes(resp[:4], byteorder="big"),
            int.from_bytes(resp[4:], byteorder="big"))


def _decode_lookup_request(
    frames: Sequence[bytes | bytearray | memoryview],
) -> tuple[int, list[bytes], int]:
    """Decode and strictly validate a five-frame lookup request."""
    if len(frames) != 5:
        raise ValueError(f"lookup request has {len(frames)} frames, expected 5")
    if len(frames[4]) != RETAIN_TTL_BYTES:
        raise ValueError(
            f"retain_ttl frame has {len(frames[4])} bytes, "
            f"expected {RETAIN_TTL_BYTES}"
        )
    retain_ttl_s = int.from_bytes(frames[4], byteorder="big")
    if bytes(frames[0]) != LOOKUP_MSG:
        raise ValueError("lookup request has an invalid message type")

    token_frame = frames[1]
    hash_len_frame = frames[2]
    if len(token_frame) != TOKEN_LEN_BYTES:
        raise ValueError(
            f"token_len frame has {len(token_frame)} bytes, "
            f"expected {TOKEN_LEN_BYTES}"
        )
    if len(hash_len_frame) != HASH_LEN_BYTES:
        raise ValueError(
            f"hash_len frame has {len(hash_len_frame)} bytes, "
            f"expected {HASH_LEN_BYTES}"
        )

    token_len = int.from_bytes(token_frame, byteorder="big")
    hash_len = int.from_bytes(hash_len_frame, byteorder="big")
    blob = memoryview(frames[3])
    if not blob:
        if hash_len != 0:
            raise ValueError("empty hash blob has a non-zero hash_len")
        return token_len, [], retain_ttl_s
    if hash_len == 0:
        raise ValueError("non-empty hash blob has zero hash_len")
    if len(blob) % hash_len:
        raise ValueError(
            f"hash blob length {len(blob)} is not divisible by hash_len {hash_len}"
        )

    return token_len, [
        bytes(blob[offset : offset + hash_len])
        for offset in range(0, len(blob), hash_len)
    ], retain_ttl_s
