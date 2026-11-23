# SPDX-License-Identifier: Apache-2.0
"""Startup checks of a connector's free-form configuration dict.

A misspelt identity key is not read: its default applies, and instances
that should be told apart share keys. So a key that is not an identity key
but nearly is fails startup, as does a non-integer value for an integer
key. Other unknown keys are left alone: the dict may carry settings of
other components.
"""

from __future__ import annotations

import difflib
from typing import Iterable, Mapping


def validate_extra_config(
    cfg: Mapping[str, object],
    *,
    identity_keys: Iterable[str],
    integer_keys: Iterable[str] = (),
    hints: Mapping[str, str] | None = None,
    label: str = "extra_config",
) -> None:
    """Refuse a near-miss of an identity key and a non-integer integer key.

    ``hints`` adds a sentence to the error of an integer key.
    """
    identity = tuple(identity_keys)
    for key in cfg:
        if key in identity:
            continue
        close = difflib.get_close_matches(str(key), identity, n=1, cutoff=0.8)
        if close:
            raise ValueError(
                f"{label} key {key!r} is not known; did you mean "
                f"{close[0]!r}? Unread, its default would apply and instances "
                f"that should be told apart would share keys")
    for key in integer_keys:
        if key not in cfg:
            continue
        value = cfg[key]
        try:
            ok = not isinstance(value, bool) and int(value) == float(value)
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            hint = (hints or {}).get(key, "")
            raise ValueError(
                f"{label} {key} must be an integer, got {value!r}{hint}")
