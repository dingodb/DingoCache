from __future__ import annotations

import re
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

if __name__ == "__main__":
    unittest.main()
