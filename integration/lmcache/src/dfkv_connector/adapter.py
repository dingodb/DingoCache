# SPDX-License-Identifier: Apache-2.0
"""LMCache RemoteConnector adapter for dfkv.

Wire-up in LMCache config (plugin mode, recommended):

    remote_storage_plugins: ["dfkv"]
    extra_config:
      remote_storage_plugin.dfkv.module_path: dfkv_connector.adapter
      remote_storage_plugin.dfkv.class_name:  DfkvConnectorAdapter
      remote_storage_plugin.dfkv.url:         dfkv://mds1:6700,mds2:6700/group-1

      # Optional knobs:
      remote_storage_plugin.dfkv.membership:  mds      # "mds" (default) | "static"
      remote_storage_plugin.dfkv.lib:         /path/to/libdfkv.so   # overrides $DFKV_LIB
      remote_storage_plugin.dfkv.mds_poll_ms: 3000
      remote_storage_plugin.dfkv.rail_affinity: true
      remote_storage_plugin.dfkv.rail_affinity_fallbacks: 1

For static membership the URL carries the member string directly, e.g.
    remote_storage_plugin.dfkv.url:        dfkv://n1=10.0.0.1:12000,n2=10.0.0.2:12000/x
    remote_storage_plugin.dfkv.membership: static

Legacy fallback: top-level ``remote_url: "dfkv://..."`` also works; per-plugin
knobs are not available on that path (defaults: membership=mds, lib from
$DFKV_LIB).

URLs MUST be ``dfkv://<endpoint>/<group>`` — no query string. Everything else
goes through extra_config above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from dfkv_common import reject_namespace_override

from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

from .access_log import access_log
from .remote_connector import DfkvConnector
from .rail_affinity import parse_plugin_rail_affinity

logger = init_logger(__name__)

_PLUGIN_TYPE = "dfkv"
_PLUGIN_URL_PREFIX = "plugin://"
_LEGACY_URL_PREFIX = "dfkv://"
_DEFAULT_MEMBERSHIP = "mds"
_DEFAULT_MDS_POLL_MS = 3000


def _extract_plugin_type(plugin_name: str) -> str:
    return plugin_name.split(".", 1)[0]


@dataclass(frozen=True)
class _PluginOverrides:
    target_url: str
    membership: str
    lib_path: Optional[str]
    mds_poll_ms: int
    rail_affinity: bool
    rail_affinity_fallbacks: int


class DfkvConnectorAdapter(ConnectorAdapter):
    """URL adapter for dfkv.

    Matches both ``dfkv://...`` (legacy ``remote_url`` path) and
    ``plugin://dfkv[.instance]`` (LMCache plugin path).
    """

    def __init__(self) -> None:
        with access_log("adapter.__init__", lambda: ""):
            super().__init__(_LEGACY_URL_PREFIX)

    def can_parse(self, url: str) -> bool:
        if url.startswith(_LEGACY_URL_PREFIX):
            return True
        if url.startswith(_PLUGIN_URL_PREFIX):
            pname = url[len(_PLUGIN_URL_PREFIX):]
            return _extract_plugin_type(pname) == _PLUGIN_TYPE
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        with access_log("adapter.create_connector",
                        lambda: f"url={context.url}"):
            ov = self._resolve_overrides(context)
            logger.info(
                "Creating DfkvConnector: context_url=%s target_url=%s "
                "membership=%s lib=%s mds_poll_ms=%d rail_affinity=%s "
                "fallbacks=%d",
                context.url, ov.target_url, ov.membership,
                ov.lib_path or "-", ov.mds_poll_ms, ov.rail_affinity,
                ov.rail_affinity_fallbacks,
            )
            return DfkvConnector(
                url=ov.target_url,
                loop=context.loop,
                local_cpu_backend=context.local_cpu_backend,
                lib_path=ov.lib_path,
                membership=ov.membership,
                mds_poll_ms=ov.mds_poll_ms,
                rail_affinity=ov.rail_affinity,
                rail_affinity_fallbacks=ov.rail_affinity_fallbacks,
            )

    @staticmethod
    def _resolve_overrides(context: ConnectorContext) -> _PluginOverrides:
        url = context.url

        if url.startswith(_LEGACY_URL_PREFIX):
            return _PluginOverrides(
                target_url=url, membership=_DEFAULT_MEMBERSHIP,
                lib_path=None, mds_poll_ms=_DEFAULT_MDS_POLL_MS,
                rail_affinity=False, rail_affinity_fallbacks=1,
            )

        plugin_name = getattr(context, "plugin_name", None) or url[
            len(_PLUGIN_URL_PREFIX):
        ]
        prefix = f"remote_storage_plugin.{plugin_name}"

        extra_config = (
            getattr(context.config, "extra_config", None)
            if context.config is not None else None
        ) or {}

        target_url = extra_config.get(f"{prefix}.url")
        if not target_url:
            raise ValueError(
                f"Plugin mode requires extra_config['{prefix}.url'] to be a "
                "dfkv:// URL (e.g. 'dfkv://mds1:6700/group-1')"
            )
        if not isinstance(target_url, str) or not target_url.startswith(
            _LEGACY_URL_PREFIX
        ):
            raise ValueError(
                f"extra_config['{prefix}.url']={target_url!r} must be a string "
                f"starting with {_LEGACY_URL_PREFIX!r}"
            )

        membership = extra_config.get(f"{prefix}.membership", _DEFAULT_MEMBERSHIP)
        if membership not in ("mds", "static"):
            raise ValueError(
                f"extra_config['{prefix}.membership']={membership!r} must be "
                "'mds' or 'static'"
            )

        lib_path = extra_config.get(f"{prefix}.lib")
        if lib_path is not None and not isinstance(lib_path, str):
            raise ValueError(
                f"extra_config['{prefix}.lib']={lib_path!r} must be a "
                "filesystem path string"
            )
        reject_namespace_override(
            extra_config, key=f"{prefix}.key_namespace")

        raw_poll = extra_config.get(f"{prefix}.mds_poll_ms")
        mds_poll_ms = _DEFAULT_MDS_POLL_MS
        if raw_poll is not None:
            try:
                mds_poll_ms = int(raw_poll)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"extra_config['{prefix}.mds_poll_ms']={raw_poll!r} must be "
                    "an integer"
                ) from e
            if mds_poll_ms <= 0:
                raise ValueError(
                    f"extra_config['{prefix}.mds_poll_ms']={mds_poll_ms} must be "
                    "positive"
                )
        affinity = parse_plugin_rail_affinity(extra_config, prefix)

        return _PluginOverrides(
            target_url=target_url,
            membership=membership,
            lib_path=lib_path or None,
            mds_poll_ms=mds_poll_ms,
            rail_affinity=affinity.enabled,
            rail_affinity_fallbacks=affinity.fallback_count,
        )
