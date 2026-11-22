#!/usr/bin/env python3
"""Idempotent, fail-safe dfkv cache-node replacement through MDS membership."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict

from dfkv_ops_common import (
    CommandError,
    RingView,
    atomic_write,
    parse_clients,
    parse_ring,
    parse_stat_health,
    run_command,
)


class WorkflowError(RuntimeError):
    pass


class Replacer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.old_stopped = False
        self.started_new = False
        # Whether the old node owned vnodes at preflight: what a rollback has
        # to bring back.
        self.old_placed = True
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
        if command == "ring":
            argv.insert(2, "--json")
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
        try:
            table = self.ctl("stat")
        except CommandError as error:
            # Exit 1 with the table is dfkvctl's verdict on the whole group:
            # some member is unreachable or unhealthy. Only one row is asked
            # about here, and the member being replaced may well be that one.
            # Exit 1 without a table is dfkvctl failing (no MDS, empty ring).
            if error.returncode != 1 or not error.stdout:
                raise
            table = error.stdout
        return parse_stat_health(table).get(node_id, False)

    def systemctl(self, host: str, *arguments: str) -> list[str]:
        return [
            self.args.ssh,
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={max(1, int(self.args.command_timeout))}",
            "--", host,
            "systemctl", *arguments, self.args.service,
        ]

    def remote(self, host: str, action: str) -> None:
        # --no-block: systemd queues the job and returns. A blocking stop
        # outlasts the command timeout, and a blocking start issued while the
        # unit is still stopping waits behind that stop; either would be
        # reported as a failure of a job that is in fact running. What the
        # job leads to is observed in the MDS view and, where the unit itself
        # matters, through wait_unit().
        command = self.systemctl(host, action, "--no-block")
        if self.args.dry_run:
            self.record("dry-run", shlex.join(command))
            return
        run_command(command, self.args.command_timeout)

    def unit_state(self, host: str) -> str:
        output = run_command(
            self.systemctl(host, "show", "--property=ActiveState"), self.args.command_timeout
        )
        name, _, state = output.strip().partition("=")
        if name != "ActiveState" or not state:
            raise ValueError(f"unexpected systemctl show output: {output!r}")
        return state

    def wait_unit(self, host: str, *states: str, timeout: float) -> None:
        self.wait_for(
            f"{self.args.service} on {host} to become {'/'.join(states)}",
            lambda: self.unit_state(host) in states,
            timeout,
        )

    def wait_for(self, description: str, predicate, timeout: float | None = None) -> object:
        timeout = self.args.timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        last_error = "condition was false"
        while time.monotonic() < deadline:
            try:
                value = predicate()
                if value:
                    return value
                last_error = "condition was false"
            # Command timeouts and transient SSH/OS failures retry like any
            # other pollable error; the deadline above still bounds total wait.
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                last_error = str(error)
            time.sleep(self.args.poll_interval)
        raise WorkflowError(f"timeout after {timeout}s waiting for {description}: {last_error}")

    def wait_stable_ring(
        self, *, placed: set[str] = frozenset(), registered: set[str] = frozenset(),
        forbidden: set[str] = frozenset(),
    ) -> RingView:
        """A view seen unchanged stable_samples times in which `placed` own
        vnodes, `registered` are members (eligible or not) and `forbidden` are
        not members at all."""
        stable = 0
        previous = ""
        final: RingView | None = None

        def sample() -> RingView | None:
            nonlocal stable, previous, final
            view = self.ring()
            members = {member.node_id for member in view.members}
            # A registered member that is not ring-eligible owns no vnodes.
            eligible = {member.node_id for member in view.members if member.eligible}
            if not (placed <= eligible and registered <= members) or forbidden & members:
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
            if not self.started_new:
                self.record("abort", f"old node was never stopped; safe abort: {reason}")
                return
            # The replacement may be running and registered in the ring even
            # when its start command timed out locally; silently aborting
            # would leave an unplanned member. Stop it best-effort, mirroring
            # the old-node restart below.
            self.record(
                "rollback",
                f"failure after replacement start; stopping {self.args.new_id} on {self.args.new_host}: {reason}",
            )
            accepted = False
            try:
                self.remote(self.args.new_host, "stop")
                accepted = True
                if not self.args.dry_run:
                    self.wait_unit(
                        self.args.new_host, "inactive", "failed", timeout=self.args.stop_timeout)
                self.record("rollback", "replacement stopped; its registration expires with its lease")
            except BaseException as rollback_error:
                unit = f"{self.args.service} on {self.args.new_host}"
                raise WorkflowError(
                    f"replacement failed ({reason}); CRITICAL stopping the just-started replacement "
                    + (
                        f"was accepted by systemd but not seen to finish ({rollback_error}); "
                        f"replacement may still be registered in the ring; check that {unit} stops"
                        if accepted else
                        f"also failed ({rollback_error}); "
                        f"replacement may still be registered in the ring; stop {unit} manually"
                    )
                ) from rollback_error
            return
        self.record("rollback", f"failure after old-node stop; restarting {self.args.old_id}: {reason}")
        accepted = False
        try:
            self.remote(self.args.old_host, "start")
            accepted = True
            if not self.args.dry_run:
                # systemd runs the start once a stop still in progress is over,
                # which can take the unit's whole stop timeout. Until then the
                # MDS can still show, and dfkvctl still reach, the process that
                # is going away; "active" is the new one.
                self.wait_unit(
                    self.args.old_host, "active",
                    timeout=self.args.stop_timeout + self.args.timeout)
                if self.old_placed:
                    self.wait_stable_ring(placed={self.args.old_id})
                    self.wait_for(
                        "old node readiness after rollback", lambda: self.healthy(self.args.old_id))
                else:
                    self.wait_stable_ring(registered={self.args.old_id})
            self.record("rollback", "old node restored; replacement left running to avoid another ring shrink")
        except BaseException as rollback_error:
            unit = f"{self.args.service} on {self.args.old_host}"
            raise WorkflowError(
                f"replacement failed ({reason}); CRITICAL rollback "
                + (
                    f"not confirmed ({rollback_error}); systemd accepted the start of {unit} and runs it "
                    f"once the unit has finished stopping, if it has not already; "
                    f"check the unit and that {self.args.old_id} rejoins the ring"
                    if accepted else
                    f"also failed ({rollback_error}); start {unit} immediately"
                )
            ) from rollback_error

    def run(self) -> None:
        initial = self.ring()
        members = {member.node_id: member for member in initial.members}
        ids = set(members)
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

        # The old node only has to be registered: one the MDS already holds
        # out of placement is a common reason for the replacement.
        self.old_placed = members[self.args.old_id].eligible
        if not self.old_placed:
            self.record(
                "preflight",
                f"{self.args.old_id} is registered but not ring-eligible; stopping it does not change placement",
            )

        if not new_present:
            self.record("join", f"starting replacement {self.args.new_id} on {self.args.new_host}")
            # Mark replacement-started before SSH: a remote systemctl may
            # succeed even when the local SSH command times out or is
            # interrupted, mirroring the old-node stop marking below.
            self.started_new = True
            self.remote(self.args.new_host, "start")
            if self.args.dry_run:
                self.record("dry-run", f"would wait up to {self.args.timeout}s for replacement membership/readiness")
                self.record("dry-run", f"would stop {self.args.old_id}, observe lease expiry, and verify clients")
                return

        joined = self.wait_stable_ring(placed={self.args.new_id}, registered={self.args.old_id})
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
        # A stopping server stops heartbeating first, so its membership
        # expires with its lease while the unit may go on flushing for
        # minutes. Nothing below depends on that having finished.
        final = self.wait_stable_ring(placed={self.args.new_id}, forbidden={self.args.old_id})
        self.wait_for("replacement readiness after old-node lease expiry", lambda: self.healthy(self.args.new_id))
        self.verify_clients()
        self.record(
            "complete",
            f"old membership expired; replacement healthy at ring epoch={final.epoch}; "
            f"{self.args.service} on {self.args.old_host} may still be finishing its stop",
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
    result.add_argument(
        "--stop-timeout", type=float, default=660.0,
        help=(
            "how long a rollback gives a unit to finish stopping, in seconds: set it above the unit's "
            "TimeoutStopSec (default: 660). A successful replacement does not wait for the old unit"
        ),
    )
    result.add_argument(
        "--command-timeout", type=float, default=10.0,
        help="timeout for each CLI/SSH command in seconds; systemctl jobs are queued, not awaited",
    )
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
    if min(args.timeout, args.stop_timeout, args.command_timeout, args.poll_interval) <= 0:
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
