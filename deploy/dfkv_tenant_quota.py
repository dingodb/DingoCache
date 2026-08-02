#!/usr/bin/env python3
"""Atomically list or edit a dfkv node's strict tenant-quota file."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

try:
    from dfkv_common.identity import tenant_hash
except ModuleNotFoundError:
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1] / "integration" / "common" / "src"),
    )
    from dfkv_common.identity import tenant_hash

from dfkv_ops_common import atomic_write

_HASH = re.compile(r"^[0-9a-f]{16}$")
_UINT64_MAX = (1 << 64) - 1


def load_quotas(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    quotas: dict[str, int] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or not _HASH.fullmatch(fields[0]):
            raise ValueError(f"malformed tenant quota line {line_number}")
        if not fields[1].isascii() or not fields[1].isdigit():
            raise ValueError(f"malformed tenant quota line {line_number}")
        limit = int(fields[1], 10)
        if limit > _UINT64_MAX:
            raise ValueError(f"malformed tenant quota line {line_number}")
        if fields[0] in quotas:
            raise ValueError(f"duplicate tenant hash on quota line {line_number}")
        quotas[fields[0]] = limit
    return quotas


def resolve_hash(args: argparse.Namespace) -> str:
    if args.hash is not None:
        if not _HASH.fullmatch(args.hash):
            raise ValueError("--hash must be exactly 16 lower-case hexadecimal characters")
        return args.hash
    if args.tenant is None or args.tenant == "":
        raise ValueError("--tenant must not be empty")
    return f"{tenant_hash(args.tenant):016x}"


def write_quotas(path: Path, quotas: dict[str, int]) -> None:
    content = "# dfkv tenant quotas: <16-lowercase-hex-hash> <uint64-bytes>\n"
    content += "".join(f"{key} {quotas[key]}\n" for key in sorted(quotas))
    atomic_write(str(path), content)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Manage the immutable-at-server-start tenant quota file. Changes take "
            "effect after node restart and never delete cached data."
        )
    )
    result.add_argument("--file", required=True, help="DFKV_TENANT_QUOTAS_FILE path")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="print strict hash/byte records")
    for name in ("set", "remove"):
        command = commands.add_parser(name)
        identity = command.add_mutually_exclusive_group(required=True)
        identity.add_argument("--tenant", help="human tenant name (UTF-8)")
        identity.add_argument("--hash", help="explicit 16-character lower-case hash")
        if name == "set":
            command.add_argument("bytes", type=int, help="uint64 byte limit; 0 means unlimited")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path = Path(args.file)
    try:
        quotas = load_quotas(path)
        if args.command == "list":
            for key in sorted(quotas):
                print(f"{key} {quotas[key]}")
            return 0
        key = resolve_hash(args)
        if args.command == "set":
            if args.bytes < 0 or args.bytes > _UINT64_MAX:
                raise ValueError("bytes must be a uint64")
            quotas[key] = args.bytes
            write_quotas(path, quotas)
            print(f"{key} {args.bytes}")
            return 0
        if key not in quotas:
            raise ValueError(f"tenant hash {key} is not configured")
        del quotas[key]
        write_quotas(path, quotas)
        print(key)
        return 0
    except (OSError, UnicodeError, ValueError) as error:
        print(f"dfkv_tenant_quota.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
