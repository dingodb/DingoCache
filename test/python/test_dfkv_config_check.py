"""Startup refusal of a misspelt identity key (dfkv_common.config_check).

The rules HiCache has applied since 2.42, shared with the vLLM connector and
the LMCache L2 adapter: a key that nearly matches an identity key fails
startup, as does a non-integer value for an integer key; unrelated keys
pass. The connectors' own tests cover their key lists.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "integration", "common", "src"))
from dfkv_common.config_check import validate_extra_config  # noqa: E402

IDENTITY = ("model_revision", "tenant_id")


class Rules(unittest.TestCase):
    def test_a_near_miss_of_an_identity_key_is_refused(self) -> None:
        for key in ("modle_revision", "model_revison", "tenantid"):
            with self.assertRaisesRegex(ValueError, "did you mean"):
                validate_extra_config({key: "x"}, identity_keys=IDENTITY)

    def test_exact_and_unrelated_keys_pass(self) -> None:
        validate_extra_config(
            {"model_revision": "x", "tenant_id": "t", "members": "a=b:1",
             "some_other_component.flag": True},
            identity_keys=IDENTITY)

    def test_integer_keys_must_hold_integers(self) -> None:
        for good in (3, "3", 3.0):
            validate_extra_config({"n": good}, identity_keys=(),
                                  integer_keys=("n",))
        for bad in ("3.5", 2.5, True, "x", None):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                validate_extra_config({"n": bad}, identity_keys=(),
                                      integer_keys=("n",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
