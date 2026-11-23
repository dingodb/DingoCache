"""Fails unless every skipped test in a gtest XML report is allowed.

Usage: check_skips.py <report.xml> <ENV_VAR>...  Each environment variable
holds whitespace-separated "Suite.Test" names that may skip. Prints every
skipped test with its reason, and fails on any failure in the report too.
"""
import os
import sys
import xml.etree.ElementTree as ET

report, sets = sys.argv[1], sys.argv[2:]
allowed = set()
for name in sets:
    allowed.update(os.environ.get(name, "").split())
root = ET.parse(report).getroot()
cases = list(root.iter("testcase"))
skipped = {}
for case in cases:
    for skip in case.iter("skipped"):
        name = case.attrib["classname"] + "." + case.attrib["name"]
        why = (skip.attrib.get("message", "") + " " + (skip.text or "")).strip()
        skipped[name] = " ".join(why.split())
print(f"{report}: {len(cases)} tests, {len(skipped)} skipped")
for name, why in sorted(skipped.items()):
    print(f"  skipped: {name} -- {why[:200]}")
assert cases, "no tests in the report"
assert not list(root.iter("failure")), "the report has failures"
unexpected = sorted(set(skipped) - allowed)
assert not unexpected, f"skipped but not allowed ({' '.join(sets)}): {unexpected}"
