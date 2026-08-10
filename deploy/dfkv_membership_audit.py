#!/usr/bin/env python3
"""Machine-readable dfkv MDS/etcd membership drift and split-brain audit."""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from dfkv_ops_common import (
    RingMember,
    RingView,
    atomic_write,
    atomic_write_with_checksum,
    parse_ring,
    run_command,
)


@dataclass(frozen=True)
class Registration:
    member: RingMember
    mod_revision: int
    create_revision: int
    lease: int


def _take_u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("truncated membership u32")
    return struct.unpack_from("=I", data, offset)[0], offset + 4


def _take_u64(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 8 > len(data):
        raise ValueError("truncated membership u64")
    return struct.unpack_from("=Q", data, offset)[0], offset + 8


def decode_registration(data: bytes) -> RingMember:
    """Decode one etcd registration value written by EncodeMembers({self}, 0)."""
    embedded_epoch, offset = _take_u64(data, 0)
    count, offset = _take_u32(data, offset)
    if embedded_epoch != 0 or count != 1:
        raise ValueError(f"registration must contain one member at epoch 0, got epoch={embedded_epoch} count={count}")

    def take_text() -> str:
        nonlocal offset
        size, offset = _take_u32(data, offset)
        if offset + size > len(data):
            raise ValueError("truncated membership string")
        try:
            value = data[offset : offset + size].decode()
        except UnicodeDecodeError as error:
            raise ValueError("membership string is not UTF-8") from error
        offset += size
        return value

    node_id = take_text()
    ip = take_text()
    port, offset = _take_u32(data, offset)
    weight, offset = _take_u32(data, offset)
    tcp_port = 0
    info = "-"
    while offset + 4 <= len(data):
        tag, offset = _take_u32(data, offset)
        if tag == 0x54435031:  # TCP1
            tcp_port, offset = _take_u32(data, offset)
        elif tag == 0x4E464F31:  # NFO1
            size, offset = _take_u32(data, offset)
            if offset + size > len(data):
                raise ValueError("truncated registration info")
            info = data[offset : offset + size].decode()
            offset += size
        elif tag == 0x31415453:  # STA1; not needed for identity comparison
            if offset >= len(data):
                raise ValueError("truncated registration stats field count")
            fields = data[offset]
            offset += 1
            if offset >= len(data):
                raise ValueError("truncated registration stats presence")
            present = data[offset] != 0
            offset += 1
            if present:
                offset += fields * 8
                if offset > len(data):
                    raise ValueError("truncated registration stats")
        else:
            break
    return RingMember(node_id, f"{ip}:{port}", weight, 0, 0.0, info)


def range_end(prefix: bytes) -> bytes:
    result = bytearray(prefix)
    for index in range(len(result) - 1, -1, -1):
        if result[index] != 0xFF:
            result[index] += 1
            return bytes(result[: index + 1])
    return b"\0"


def read_registrations(etcd: str, group: str, timeout: float) -> tuple[int, list[Registration]]:
    base = etcd if "://" in etcd else "http://" + etcd
    prefix = f"/dfkv/v1/groups/{group}/members/".encode()
    body = json.dumps(
        {
            "key": base64.b64encode(prefix).decode(),
            "range_end": base64.b64encode(range_end(prefix)).decode(),
        }
    ).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/v3/kv/range",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"etcd range failed: {error}") from error
    revision = int(payload.get("header", {}).get("revision", 0))
    registrations: list[Registration] = []
    for raw in payload.get("kvs", []):
        key = base64.b64decode(raw["key"]).decode()
        member = decode_registration(base64.b64decode(raw["value"]))
        expected_id = key.rsplit("/", 1)[-1]
        if expected_id != member.node_id:
            raise ValueError(f"etcd key ID {expected_id!r} differs from self-reported ID {member.node_id!r}")
        registrations.append(
            Registration(member, int(raw.get("mod_revision", 0)), int(raw.get("create_revision", 0)), int(raw.get("lease", 0)))
        )
    return revision, registrations


def info_fields(info: str) -> dict[str, str]:
    if info == "-":
        return {}
    result: dict[str, str] = {}
    for part in info.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key] = value
    return result


def core(view: RingView) -> list[tuple[str, str, int]]:
    return sorted((member.node_id, member.address, member.weight) for member in view.members)


def audit(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    issues: list[dict[str, str]] = []
    views: dict[str, RingView] = {}
    for endpoint in [item for item in args.mds.split(",") if item]:
        try:
            command = [
                args.dfkvctl, "ring", "--mds", endpoint,
                "--group", args.group,
            ]
            if args.allow_empty:
                command.append("--allow-empty")
            output = run_command(command, args.timeout)
            views[endpoint] = parse_ring(output)
        except (RuntimeError, ValueError) as error:
            issues.append({"severity": "critical", "code": "mds_unreachable", "endpoint": endpoint, "message": str(error)})

    etcd_revision = 0
    registrations: list[Registration] = []
    try:
        etcd_revision, registrations = read_registrations(args.etcd, args.group, args.timeout)
    except (RuntimeError, ValueError) as error:
        issues.append({"severity": "critical", "code": "etcd_unreachable_or_invalid", "message": str(error)})

    canonical: RingView | None = next(iter(views.values()), None)
    if canonical is not None:
        if not canonical.members and not args.allow_empty:
            issues.append(
                {
                    "severity": "critical",
                    "code": "empty_ring",
                    "message": "MDS and etcd expose no cache members; use --allow-empty only for an intentionally inactive group",
                }
            )
        for endpoint, view in views.items():
            if core(view) != core(canonical) or view.epoch != canonical.epoch:
                issues.append(
                    {
                        "severity": "critical",
                        "code": "mds_split_brain",
                        "endpoint": endpoint,
                        "message": "MDS endpoint returned a different placement view/ring epoch",
                    }
                )

        reg_by_id = {registration.member.node_id: registration for registration in registrations}
        mds_by_id = {member.node_id: member for member in canonical.members}
        missing_etcd = sorted(set(mds_by_id) - set(reg_by_id))
        missing_mds = sorted(set(reg_by_id) - set(mds_by_id))
        if missing_etcd or missing_mds:
            issues.append(
                {
                    "severity": "critical",
                    "code": "mds_etcd_membership_drift",
                    "message": f"only_mds={missing_etcd} only_etcd={missing_mds}",
                }
            )
        for node_id in sorted(set(mds_by_id) & set(reg_by_id)):
            mds_member = mds_by_id[node_id]
            registration = reg_by_id[node_id].member
            if (mds_member.address, mds_member.weight, mds_member.info) != (
                registration.address,
                registration.weight,
                registration.info,
            ):
                issues.append(
                    {
                        "severity": "critical",
                        "code": "registration_self_report_mismatch",
                        "node": node_id,
                        "message": "MDS view differs from authoritative etcd registration value",
                    }
                )
            fields = info_fields(registration.info)
            absent = [name for name in args.require_info_field if not fields.get(name)]
            if absent:
                issues.append(
                    {
                        "severity": "warning" if args.allow_missing_info else "critical",
                        "code": "self_report_incomplete",
                        "node": node_id,
                        "message": "missing INFO fields: " + ",".join(absent),
                    }
                )

    has_critical = any(issue["severity"] == "critical" for issue in issues)
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": int(time.time()),
        "group": args.group,
        "healthy": not has_critical,
        "ring_epoch": canonical.epoch if canonical else None,
        "etcd_header_revision": etcd_revision,
        "mds_views": {
            endpoint: {
                "ring_epoch": view.epoch,
                "ring_points": view.ring_points,
                "members": [asdict(member) for member in view.members],
            }
            for endpoint, view in views.items()
        },
        "registrations": [
            {
                "node_id": registration.member.node_id,
                "address": registration.member.address,
                "weight": registration.member.weight,
                "info": registration.member.info,
                "create_revision": registration.create_revision,
                "registration_revision": registration.mod_revision,
                "lease": registration.lease,
            }
            for registration in sorted(registrations, key=lambda item: item.member.node_id)
        ],
        "issues": issues,
    }
    return report, 2 if has_critical else 0


def prometheus(report: dict[str, object]) -> str:
    group = str(report["group"]).replace("\\", "\\\\").replace('"', '\\"')
    issues = report["issues"]
    assert isinstance(issues, list)
    critical = sum(1 for issue in issues if isinstance(issue, dict) and issue.get("severity") == "critical")
    warning = sum(1 for issue in issues if isinstance(issue, dict) and issue.get("severity") == "warning")
    lines = [
        "# HELP dfkv_membership_audit_ok Whether the last MDS/etcd membership audit found no drift (1=yes)",
        "# TYPE dfkv_membership_audit_ok gauge",
        f'dfkv_membership_audit_ok{{group="{group}"}} {1 if report["healthy"] else 0}',
        "# HELP dfkv_membership_audit_issues Issues found by severity in the last membership audit",
        "# TYPE dfkv_membership_audit_issues gauge",
        f'dfkv_membership_audit_issues{{group="{group}",severity="critical"}} {critical}',
        f'dfkv_membership_audit_issues{{group="{group}",severity="warning"}} {warning}',
        "# HELP dfkv_membership_ring_epoch Content hash epoch of the audited placement ring",
        "# TYPE dfkv_membership_ring_epoch gauge",
    ]
    if report["ring_epoch"] is not None:
        lines.append(f'dfkv_membership_ring_epoch{{group="{group}"}} {report["ring_epoch"]}')
    lines.extend(
        [
            "# HELP dfkv_membership_registration_revision Etcd mod_revision of a node registration",
            "# TYPE dfkv_membership_registration_revision gauge",
        ]
    )
    registrations = report["registrations"]
    assert isinstance(registrations, list)
    for item in registrations:
        assert isinstance(item, dict)
        node = str(item["node_id"]).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'dfkv_membership_registration_revision{{group="{group}",node="{node}"}} {item["registration_revision"]}'
        )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Detect dfkv membership drift/split brain by querying every MDS separately, "
            "decoding authoritative etcd registrations, validating node INFO self-reports, "
            "and comparing calculated content-hash ring epochs. Exit 2 means drift."
        )
    )
    result.add_argument("--mds", required=True, help="comma-separated MDS endpoints (each is audited independently)")
    result.add_argument("--etcd", required=True, help="etcd gRPC-gateway endpoint, e.g. http://127.0.0.1:2379")
    result.add_argument("--group", default="default", help="MDS group (default: default)")
    result.add_argument("--dfkvctl", default="dfkvctl", help="dfkvctl executable")
    result.add_argument("--timeout", type=float, default=5.0, help="timeout per MDS/etcd request in seconds")
    result.add_argument("--output", help="write JSON report atomically (stdout when omitted)")
    result.add_argument("--prom-output", help="write Prometheus textfile-collector metrics atomically")
    result.add_argument(
        "--require-info-field",
        action="append",
        default=[],
        metavar="NAME",
        help="required node self-report INFO field (repeatable; defaults to ver,engine,disks,cap,ram,rdma)",
    )
    result.add_argument("--allow-missing-info", action="store_true", help="downgrade missing self-report fields to warnings")
    result.add_argument("--allow-empty", action="store_true", help="accept an intentionally inactive group with no members")
    return result

def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.timeout <= 0:
        parser().error("--timeout must be positive")
    if not any(item for item in args.mds.split(",") if item):
        parser().error("--mds must contain at least one endpoint")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.group):
        parser().error("invalid group")
    if not args.require_info_field:
        args.require_info_field = ["ver", "engine", "disks", "cap", "ram", "rdma"]
    try:
        report, status = audit(args)
        content = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            atomic_write_with_checksum(args.output, content)
        else:
            sys.stdout.write(content)
        if args.prom_output:
            atomic_write(args.prom_output, prometheus(report))
        return status
    except BaseException as error:
        print(f"ERROR: audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
