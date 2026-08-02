#!/usr/bin/env python3
"""Idempotent, fail-safe dfkv cache-node replacement through MDS membership."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from dataclasses import asdict

from dfkv_ops_common import (
    RingView,
    atomic_write,
    parse_clients,
    parse_ring,
    parse_stat_reachability,
    run_command,
)


class WorkflowError(RuntimeError):
    pass


class Replacer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.old_stopped = False
        self.started_new = False
        self.initial_clients: set[str] = set()
        self.events: list[dict[str, object]] = []

    def record(self, phase: str, message: str, **fields: object) -> None:
        event = {"time": int(time.time()), "phase": phase, "message": message, **fields}
        self.events.append(event)
        print(f"[{phase}] {message}", flush=True)
        if self.args.state_file:
            atomic_write(self.args.state_file, json.dumps({"events": self.events}, indent=2) + "\n")

    def ctl(self, command: str) -> str:
        argv = [self.args.dfkvctl, command, "--mds", self.args.mds, "--group", self.args.group]
        if command == "stat":
            argv.insert(2, "--all")
        return run_command(argv, self.args.command_timeout)

    def ring(self) -> RingView:
        view = parse_ring(self.ctl("ring"))
        if view.group != self.args.group:
            raise WorkflowError(f"dfkvctl returned group {view.group!r}, expected {self.args.group!r}")
        return view

    def clients(self) -> set[str]:
        group, clients = parse_clients(self.ctl("clients"))
        if group != self.args.group:
            raise WorkflowError(f"dfkvctl returned client group {group!r}")
        return set(clients)

    def healthy(self, node_id: str) -> bool:
        return parse_stat_reachability(self.ctl("stat")).get(node_id, False)

    def remote(self, host: str, action: str) -> None:
        command = [
            self.args.ssh,
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={max(1, int(self.args.command_timeout))}",
            "--", host,
            "systemctl", action, self.args.service,
        ]
        if self.args.dry_run:
            self.record("dry-run", shlex.join(command))
            return
        run_command(command, self.args.command_timeout)

    def wait_for(self, description: str, predicate) -> object:
        deadline = time.monotonic() + self.args.timeout
        last_error = "condition was false"
        while time.monotonic() < deadline:
            try:
                value = predicate()
                if value:
                    return value
                last_error = "condition was false"
            except (RuntimeError, ValueError) as error:
                last_error = str(error)
            time.sleep(self.args.poll_interval)
        raise WorkflowError(f"timeout after {self.args.timeout}s waiting for {description}: {last_error}")

    def wait_stable_ring(self, required: set[str], forbidden: set[str]) -> RingView:
        stable = 0
        previous = ""
        final: RingView | None = None

        def sample() -> RingView | None:
            nonlocal stable, previous, final
            view = self.ring()
            ids = {member.node_id for member in view.members}
            if not required.issubset(ids) or forbidden.intersection(ids):
                stable = 0
                previous = ""
                return None
            fingerprint = view.fingerprint()
            stable = stable + 1 if fingerprint == previous else 1
            previous = fingerprint
            final = view
            return view if stable >= self.args.stable_samples else None

        result = self.wait_for("a stable MDS ring", sample)
        assert isinstance(result, RingView)
        return result

    def verify_clients(self) -> None:
        current = self.clients()
        missing = self.initial_clients - current
        if missing:
            raise WorkflowError("registered clients disappeared during replacement: " + ",".join(sorted(missing)))
        if len(current) < self.args.min_clients:
            raise WorkflowError(f"only {len(current)} registered clients; require {self.args.min_clients}")

    def rollback(self, reason: BaseException) -> None:
        if not self.old_stopped:
            self.record("abort", f"old node was never stopped; safe abort: {reason}")
            return
        self.record("rollback", f"failure after old-node stop; restarting {self.args.old_id}: {reason}")
        try:
            self.remote(self.args.old_host, "start")
            if not self.args.dry_run:
                self.wait_stable_ring({self.args.old_id}, set())
                self.wait_for("old node readiness after rollback", lambda: self.healthy(self.args.old_id))
            self.record("rollback", "old node restored; replacement left running to avoid another ring shrink")
        except BaseException as rollback_error:
            raise WorkflowError(
                f"replacement failed ({reason}); CRITICAL rollback also failed ({rollback_error}); "
                f"start {self.args.service} on {self.args.old_host} immediately"
            ) from rollback_error

    def run(self) -> None:
        initial = self.ring()
        ids = {member.node_id for member in initial.members}
        self.initial_clients = self.clients()
        if len(self.initial_clients) < self.args.min_clients:
            raise WorkflowError(
                f"preflight has {len(self.initial_clients)} registered clients; require {self.args.min_clients}"
            )
        self.record(
            "preflight",
            f"ring epoch={initial.epoch} members={len(ids)} clients={len(self.initial_clients)}",
            ring_epoch=initial.epoch,
            members=sorted(ids),
            clients=sorted(self.initial_clients),
        )

        old_present = self.args.old_id in ids
        new_present = self.args.new_id in ids
        if not old_present:
            if new_present and self.healthy(self.args.new_id):
                self.verify_clients()
                self.record("complete", "old node absent and replacement healthy; nothing to do")
                return
            raise WorkflowError("old node is absent but replacement is not healthy; refusing ambiguous mutation")

        if not new_present:
            self.record("join", f"starting replacement {self.args.new_id} on {self.args.new_host}")
            self.remote(self.args.new_host, "start")
            self.started_new = True
            if self.args.dry_run:
                self.record("dry-run", f"would wait up to {self.args.timeout}s for replacement membership/readiness")
                self.record("dry-run", f"would stop {self.args.old_id}, observe lease expiry, and verify clients")
                return

        joined = self.wait_stable_ring({self.args.old_id, self.args.new_id}, set())
        self.wait_for("replacement node readiness", lambda: self.healthy(self.args.new_id))
        self.verify_clients()
        self.record("join", f"replacement ready in stable ring epoch={joined.epoch}", ring_epoch=joined.epoch)
        if self.args.dry_run:
            self.record("dry-run", f"would stop {self.args.old_id}, wait for lease expiry, and recheck readiness/clients")
            return


        self.record("drain", f"stopping old node {self.args.old_id} on {self.args.old_host}")
        # Mark rollback-required before SSH: a remote systemctl may succeed even
        # when the local SSH command times out or is interrupted.
        self.old_stopped = True
        self.remote(self.args.old_host, "stop")
        final = self.wait_stable_ring({self.args.new_id}, {self.args.old_id})
        self.wait_for("replacement readiness after old-node lease expiry", lambda: self.healthy(self.args.new_id))
        self.verify_clients()
        self.record(
            "complete",
            f"old membership expired; replacement healthy at ring epoch={final.epoch}",
            ring_epoch=final.epoch,
            members=[member.node_id for member in final.members],
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Safely replace one dfkv cache node: pre-join and health-check the replacement, "
            "observe a stable MDS/client view, stop the old service, then wait for lease expiry. "
            "A post-stop failure restarts the old node before exiting nonzero."
        )
    )
    result.add_argument("--mds", required=True, help="comma-separated dfkv_mds endpoints")
    result.add_argument("--group", default="default", help="MDS group (default: default)")
    result.add_argument("--old-id", required=True, help="current MDS member ID")
    result.add_argument("--new-id", required=True, help="replacement MDS member ID (must differ)")
    result.add_argument("--old-host", required=True, help="SSH target hosting the old service")
    result.add_argument("--new-host", required=True, help="SSH target hosting the replacement service")
    result.add_argument("--service", default="dfkv", help="systemd unit name (default: dfkv)")
    result.add_argument("--dfkvctl", default="dfkvctl", help="dfkvctl executable")
    result.add_argument("--ssh", default="ssh", help="SSH executable")
    result.add_argument("--timeout", type=float, default=180.0, help="timeout for each observed transition in seconds")
    result.add_argument("--command-timeout", type=float, default=10.0, help="timeout for each CLI/SSH command in seconds")
    result.add_argument("--poll-interval", type=float, default=3.0, help="observation interval in seconds")
    result.add_argument("--stable-samples", type=int, default=2, help="identical consecutive ring views required")
    result.add_argument("--min-clients", type=int, default=0, help="minimum registered clients required throughout")
    result.add_argument("--state-file", help="atomically updated JSON event journal")
    result.add_argument("--dry-run", action="store_true", help="inspect preconditions and print mutations without executing them")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.old_id == args.new_id:
        parser().error("--old-id and --new-id must differ (duplicate IDs can steal an etcd lease)")
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", args.old_host + args.new_host):
        parser().error("SSH hosts contain unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.service):
        parser().error("invalid systemd service name")
    if args.timeout <= 0 or args.command_timeout <= 0 or args.poll_interval <= 0:
        parser().error("timeouts and poll interval must be positive")
    if args.stable_samples < 1 or args.min_clients < 0:
        parser().error("stable samples must be >=1 and min clients must be >=0")
    workflow = Replacer(args)
    try:
        workflow.run()
        return 0
    except KeyboardInterrupt as error:
        try:
            workflow.rollback(error)
        except WorkflowError as critical:
            print(f"ERROR: {critical}", file=sys.stderr)
        return 130
    except BaseException as error:
        try:
            workflow.rollback(error)
        except WorkflowError as critical:
            print(f"ERROR: {critical}", file=sys.stderr)
            return 4
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
