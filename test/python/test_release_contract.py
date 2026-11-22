from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def project_version(path: Path) -> str:
    in_project = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_project = line == "[project]"
        elif in_project:
            match = re.fullmatch(r'version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', line)
            if match:
                return match.group(1)
    raise AssertionError(f"{path} has no static [project] version")


class ReleaseVersionContractTest(unittest.TestCase):
    def test_connector_packages_match_root_version(self):
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        manifests = [
            ROOT / "integration" / "common" / "pyproject.toml",
            ROOT / "integration" / "lmcache" / "pyproject.toml",
            ROOT / "integration" / "vllm" / "pyproject.toml",
        ]
        for manifest in manifests:
            self.assertEqual(project_version(manifest), version, manifest)

        required = f'"dfkv-common=={version}"'
        for manifest in manifests[1:]:
            self.assertIn(required, manifest.read_text(encoding="utf-8"), manifest)

    def test_changelog_has_a_section_for_the_current_version(self):
        # The commit that bumps VERSION also closes "## Unreleased": it puts
        # a "## <version> — <date>" heading above the entries the release
        # ships and leaves "## Unreleased" on top for what comes next.
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        headings = [line for line in changelog.splitlines() if line.startswith("## ")]
        self.assertEqual(headings[0], "## Unreleased")
        self.assertRegex(
            headings[1],
            rf"^## {re.escape(version)} — \d{{4}}-\d{{2}}-\d{{2}}$",
            "CHANGELOG.md: the first heading after '## Unreleased' must be "
            f"'## {version} — <YYYY-MM-DD>' (VERSION is {version})",
        )

    def test_release_workflow_checks_tag_version(self):
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('test "$tag" = "v$version"', workflow)
        self.assertIn("gh release create", workflow)
        self.assertIn("sha256sum", workflow)
        self.assertIn("docker build --target runtime", workflow)
        self.assertIn("docker image save", workflow)
        self.assertIn("ctest --test-dir build-release --output-on-failure", workflow)
        self.assertIn('git merge-base --is-ancestor "$GITHUB_SHA" origin/main', workflow)
        self.assertIn("--exclude-regex '^RdmaLoopback\\.'", workflow)
        self.assertIn("promtool", workflow)
        self.assertIn("--no-deps --target wheel-check release/*.whl", workflow)
        self.assertIn("dfkv_connector/_telemetry/metrics_push.py", workflow)
        self.assertIn("dfkv_vllm/_telemetry/metrics_push.py", workflow)
        self.assertIn("$root/test/python", workflow)


    def test_workflows_pin_actions_and_publish_once(self):
        for name in ("ci.yml", "release.yml"):
            workflow = (ROOT / ".github" / "workflows" / name).read_text(
                encoding="utf-8"
            )
            refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", workflow)
            self.assertTrue(refs, name)
            for ref in refs:
                self.assertRegex(ref, r"^[0-9a-f]{40}$", (name, ref))

        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        portable = ci.split("  portable-static-build:", 1)[1].split(
            "\n  python-wheels:", 1
        )[0]
        self.assertNotIn("actions/setup-python", portable)
        self.assertIn("python3 python3-pip", portable)

        release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("persist-credentials: false", release)
        self.assertIn("needs: build-test-package", release)
        self.assertIn("refusing to replace immutable assets", release)
        self.assertNotIn("gh release upload \"$tag\" release/* --clobber\n"
                         "          else", release)
    def _pull_images(self, failing_hosts):
        """Runs the workflow pull script against a stand-in `docker` that
        fails every pull from `failing_hosts`; returns (rc, docker calls)."""
        script = ROOT / ".github" / "scripts" / "pull_observability_images.sh"
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "calls"
            log.touch()
            fake = Path(temp) / "docker"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                'echo "$*" >> "$FAKE_LOG"\n'
                'if [ "$1" = pull ]; then\n'
                '  for h in $FAKE_FAIL; do case "$2" in "$h"/*) exit 1;; esac; done\n'
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = dict(
                os.environ,
                PATH=f"{temp}{os.pathsep}{os.environ['PATH']}",
                FAKE_LOG=str(log),
                FAKE_FAIL=" ".join(failing_hosts),
                DFKV_CI_PULL_BACKOFF_S="0",
            )
            env.pop("DFKV_CI_IMAGE_SOURCES", None)
            result = subprocess.run(
                ["bash", str(script)], cwd=ROOT, env=env, capture_output=True,
                text=True, timeout=60, check=False,
            )
            return result.returncode, log.read_text(encoding="utf-8").splitlines()

    def test_observability_image_pulls_retry_and_fall_back(self):
        compose = (
            ROOT / "deploy" / "observability" / "docker-compose.yml"
        ).read_text(encoding="utf-8")
        pinned = {
            name: re.search(rf"/({re.escape(name)}@sha256:[0-9a-f]{{64}})", compose).group(1)
            for name in ("prom/prometheus", "otel/opentelemetry-collector-contrib",
                         "grafana/tempo")
        }
        mirror, canonical = "docker.m.daocloud.io", "docker.io"

        # All healthy: one pull per image, from the canonical registry, by
        # digest.
        rc, calls = self._pull_images([])
        self.assertEqual(rc, 0)
        self.assertEqual(
            [call for call in calls if call.startswith("pull ")],
            [f"pull {canonical}/{ref}" for ref in pinned.values()],
        )
        self.assertEqual(
            [call.split()[-1] for call in calls if call.startswith("tag ")],
            ["dfkv-ci/prometheus:ci", "dfkv-ci/otel-collector:ci", "dfkv-ci/tempo:ci"],
        )

        # Canonical registry down: each image is retried there, then comes
        # from the mirror under the same digest.
        rc, calls = self._pull_images([canonical])
        self.assertEqual(rc, 0)
        for ref in pinned.values():
            self.assertEqual(calls.count(f"pull {canonical}/{ref}"), 3, ref)
            self.assertEqual(calls.count(f"pull {mirror}/{ref}"), 1, ref)
        self.assertEqual(sum(call.startswith("tag ") for call in calls), 3)

        # Everything down: bounded, and a failure, not a silent pass.
        rc, calls = self._pull_images([mirror, canonical])
        self.assertNotEqual(rc, 0)
        self.assertEqual(len(calls), 5)

        # Both workflows validate with what the script pulled, and neither
        # runs an image straight from a registry.
        for name in ("ci.yml", "release.yml"):
            workflow = (ROOT / ".github" / "workflows" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn("bash .github/scripts/pull_observability_images.sh", workflow)
            for tag in ("dfkv-ci/prometheus:ci", "dfkv-ci/otel-collector:ci",
                        "dfkv-ci/tempo:ci"):
                self.assertIn(tag, workflow, name)
            self.assertNotIn("docker.m.daocloud.io", workflow, name)
            self.assertNotIn("docker pull", workflow, name)

if __name__ == "__main__":
    unittest.main()
