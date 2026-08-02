# SPDX-License-Identifier: Apache-2.0
"""URL parsing for the dfkv LMCache connector.

URL grammar is intentionally minimal:

    dfkv://<endpoint>/<group>

where <endpoint> is interpreted by the membership mode (passed in from the
adapter via extra_config, default "mds"):

  - membership="mds"    (default): <endpoint> is a comma-separated MDS host:port
                                   list stored in dfkv_client_options_v2; discovery
                                   starts as part of dfkv_open_v2().
  - membership="static":           <endpoint> is a literal dfkv member string
                                   "name=ip:port,name2=ip:port2" stored in the same
                                   options descriptor; <group> is unused.

Anything else — the libdfkv.so path, MDS poll interval, connector knobs — goes
through the LMCache yaml's ``extra_config`` (see adapter.py). URLs carrying a
query string (``?k=v``) are rejected with a message pointing at extra_config.
"""

from __future__ import annotations

from dataclasses import dataclass

from .access_log import access_log

__all__ = ["DfkvEndpoint", "parse_dfkv_url"]


@dataclass(frozen=True)
class DfkvEndpoint:
    raw_endpoint: str   # MDS "host:port,host:port"  OR  members "n=ip:port,..."
    group: str
    membership: str     # "mds" | "static"


_SCHEME = "dfkv://"
_MODES = ("mds", "static")


def parse_dfkv_url(url: str, membership: str = "mds") -> DfkvEndpoint:
    with access_log("parse_dfkv_url", lambda: f"{url} ({membership})"):
        if membership not in _MODES:
            raise ValueError(
                f"membership must be one of {_MODES}, got {membership!r}"
            )
        if not url.startswith(_SCHEME):
            raise ValueError(f"not a dfkv:// URL: {url}")

        body = url[len(_SCHEME):]
        if "?" in body:
            raise ValueError(
                "dfkv URL does not accept a query string. Put the libdfkv.so "
                "path in 'remote_storage_plugin.dfkv.lib', the membership mode "
                "in 'remote_storage_plugin.dfkv.membership', and other knobs in "
                f"'remote_storage_plugin.dfkv.<knob>' (or env). Got: {url!r}"
            )

        endpoint, _, group = body.partition("/")
        if not endpoint:
            raise ValueError(f"missing <endpoint> in {url}")

        if membership == "mds":
            if not group:
                raise ValueError(
                    f"mds membership requires dfkv://<mds_addrs>/<group>; "
                    f"missing /<group> in {url}"
                )
        else:  # static
            if "=" not in endpoint:
                raise ValueError(
                    "static membership requires <endpoint> to be a member "
                    "string 'name=ip:port,name2=ip:port2'; "
                    f"got {endpoint!r} in {url}"
                )

        return DfkvEndpoint(
            raw_endpoint=endpoint, group=group, membership=membership
        )
