#!/usr/bin/env python3
"""Real-CUDA regression for hybrid HiCache state restoration, no model weights.

Run inside the supported SGLang image with the dfkv checkout at /dfkv:
  python3 /dfkv/integration/hicache/tests/gpu_hybrid_state_roundtrip.py \
      --include-scaled-mla
An explicit --assembler-override /path/to/hybrid_pool_assembler.py permits a
candidate comparison without modifying the installed SGLang package.

Exit 0: every requested device-byte comparison passed. Exit 1: data mismatch.
Exit 2: unavailable capability/setup/transfer error. Exit 124: watchdog timeout.
Stdout contains one JSON document; library diagnostics go to stderr.
This exercises actual L1/L2 pools, allocators, controller and CUDA transfers;
it does not replace model, all-rank, L3, MTP or speculative-state acceptance.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from types import SimpleNamespace


def digest(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def forbid_eviction(*args, **kwargs):
    raise RuntimeError("Tiny fixture unexpectedly requested cache eviction")


def run_case(torch, assembler, case, server_args):
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
        Mamba2StateDType,
    )
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        HybridLinearKVPool,
        MambaPool,
    )
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    page_size, token_capacity, slots = 64, 256, 4
    full_layers, linear_layers = [1, 3], [0, 2]
    kv_dim = 656 if case["layout"] == "scaled-mla" else 576
    case["phase"] = "construct_device_pools"
    mamba = MambaPool(
        size=slots,
        spec_state_size=0,
        cache_params=KimiLinearCacheParams(
            shape=KimiLinearStateShape.create(
                tp_world_size=1, num_heads=2, head_dim=16, conv_kernel_size=4
            ),
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=linear_layers,
        ),
        mamba_layer_ids=linear_layers,
        device="cuda",
    )
    hybrid = HybridLinearKVPool(
        size=token_capacity,
        dtype=torch.float8_e4m3fn,
        page_size=page_size,
        head_num=1,
        head_dim=576,
        full_attention_layer_ids=full_layers,
        device="cuda",
        mamba_pool=mamba,
        use_mla=True,
        use_dsa=True,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        kv_cache_dim=kv_dim,
    )
    kv = hybrid.full_kv_pool
    require(isinstance(kv, DSATokenToKVPool), "Hybrid did not construct a real DSA pool")
    require(kv.kv_cache_dim == kv_dim, "Requested device MLA geometry was not honored")
    require(kv.dsa_kv_cache_store_fp8 == (kv_dim == 656), "Unexpected DSA FP8 storage mode")
    kv_allocator = PagedTokenToKVPoolAllocator(
        token_capacity, page_size, torch.float8_e4m3fn, "cuda", hybrid, False
    )
    mamba_allocator = MambaSlotAllocator(slots, "cuda")
    # Request/tree metadata only. Pools, slot allocation and all transfers are native.
    req_pool = SimpleNamespace(
        mamba_pool=mamba,
        mamba_allocator=mamba_allocator,
        mamba_map={global_id: local_id for local_id, global_id in enumerate(linear_layers)},
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=kv_allocator,
        page_size=page_size,
    )
    cache = SimpleNamespace(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=kv_allocator,
        evict_host=forbid_eviction,
        evict_for_alloc=forbid_eviction,
    )
    case["phase"] = "assemble_native_strategy"
    strategy = assembler._MambaStrategy()
    require(
        strategy.matches(hybrid, {ComponentType.FULL, ComponentType.MAMBA}),
        "Native hybrid strategy does not match actual pools",
    )
    stack = strategy.build(
        cache=cache,
        kvcache=hybrid,
        params=params,
        server_args=server_args,
        load_cache_event=threading.Event(),
        storage_backend=None,
    )
    group, controller = stack.host_pool_group, stack.cache_controller
    case["assembled_pools"] = [entry.name.value for entry in group.entries]
    case["sidecars"] = [
        {"pool": spec.pool_name.value, "indices_from_pool": spec.indices_from_pool.value}
        for spec in stack.sidecars
    ]
    case["geometry"] = {
        "page_size": page_size,
        "token_capacity": token_capacity,
        "mamba_slot_capacity": slots,
        "device_kv_dim": kv.kv_cache_dim,
        "host_kv_dim": group.get_pool(PoolName.KV).kv_cache_dim,
        "full_layer_mapping": hybrid.full_attention_layer_id_mapping,
        "mamba_layer_mapping": req_pool.mamba_map,
        "transfer_layer_num": stack.transfer_layer_num,
    }
    source_kv = kv_allocator.alloc(2 * page_size)
    source_mamba = mamba_allocator.alloc(2)
    require(source_kv is not None and source_mamba is not None, "Source allocation failed")
    source_pages = source_kv.reshape(-1, page_size)[:, 0] // page_size
    source_indices = {"kv": source_kv, "indexer": source_pages, "mamba": source_mamba}
    buffers = []
    for local_id, global_id in enumerate(full_layers):
        buffers.append((f"kv.layer_{global_id}", "kv", kv.kv_buffer[local_id]))
        buffers.append((f"indexer.layer_{global_id}", "indexer", kv.index_k_with_scale_buffer[local_id]))
    for local_id, global_id in enumerate(linear_layers):
        buffers.append((f"temporal.layer_{global_id}", "mamba", mamba.mamba_cache.temporal[local_id]))
        for conv_id, conv in enumerate(mamba.mamba_cache.conv):
            buffers.append((f"conv_{conv_id}.layer_{global_id}", "mamba", conv[local_id]))
    case["device_state_allocation_bytes"] = sum(t.numel() * t.element_size() for _, _, t in buffers)
    require(case["device_state_allocation_bytes"] < 4 * 1024**2, "Fixture exceeded 4 MiB state bound")
    expected = {}
    case["phase"] = "fill_known_bytes"
    for ordinal, (name, kind, tensor) in enumerate(buffers):
        require(tensor.is_cuda and tensor.is_contiguous(), f"Unsupported physical buffer: {name}")
        raw = tensor.view(torch.uint8).reshape(tensor.shape[0], -1)
        # 255 is reserved for poison. Patterns differ across states/layers/rows.
        pattern = ((torch.arange(raw.numel(), device="cuda", dtype=torch.int64) * 17 + ordinal * 29) % 251).to(torch.uint8)
        raw.copy_(pattern.reshape_as(raw))
        expected[name] = raw.index_select(0, source_indices[kind]).cpu()
        case["states"][name] = {
            "status": "not_restored",
            "bytes": expected[name].numel(),
            "expected_sha256": digest(expected[name]),
            "dtype": str(tensor.dtype),
            "buffer_shape": list(tensor.shape),
        }
    torch.cuda.synchronize()

    # Follow the strategy's declared sidecars exactly. In the original module
    # INDEXER is absent, so it is not copied; its *data comparison* must fail.
    backup_extra = [PoolTransfer(name=PoolName.MAMBA, device_indices=source_mamba)]
    backup_extra.extend(
        PoolTransfer(name=spec.pool_name, indices_from_pool=spec.indices_from_pool, hit_policy=spec.hit_policy)
        for spec in stack.sidecars
    )
    case["phase"] = "offload_device_to_host"
    host_kv = controller.write(source_kv, extra_pools=backup_extra)
    require(host_kv is not None, "Host allocation failed")
    controller.ack_write_queue[-1].finish_event.synchronize()
    case["offload_completed"] = True
    case["phase"] = "poison_all_device_state"
    for name, kind, tensor in buffers:
        raw = tensor.view(torch.uint8).reshape(tensor.shape[0], -1)
        raw.fill_(255)
        poisoned = raw.cpu()
        require(bool(torch.all(poisoned == 255)), f"Poison did not reach actual device buffer: {name}")
        case["states"][name]["poison_verified"] = True
    torch.cuda.synchronize()

    restore_extra = [
        PoolTransfer(
            name=transfer.name,
            host_indices=transfer.host_indices,
            indices_from_pool=transfer.indices_from_pool,
            hit_policy=transfer.hit_policy,
        )
        for transfer in backup_extra
    ]
    case["phase"] = "restore_host_to_new_device_slots"
    destination_kv = controller.load(host_kv, extra_pools=restore_extra)
    require(destination_kv is not None, "Restore device allocation failed")
    destination_mamba = restore_extra[0].device_indices
    require(destination_mamba is not None, "Restore Mamba allocation failed")
    require(not bool(torch.isin(destination_kv, source_kv).any()), "Restore reused occupied source KV")
    require(not bool(torch.isin(destination_mamba, source_mamba).any()), "Restore reused occupied source Mamba")
    controller.start_loading()
    controller.ack_load_queue[-1].finish_event.synchronize()
    case["restore_completed"] = True
    destination_indices = {
        "kv": destination_kv,
        "indexer": destination_kv.reshape(-1, page_size)[:, 0] // page_size,
        "mamba": destination_mamba,
    }
    case["source_pages"] = source_pages.cpu().tolist()
    case["destination_pages"] = destination_indices["indexer"].cpu().tolist()
    case["source_mamba_slots"] = source_mamba.cpu().tolist()
    case["destination_mamba_slots"] = destination_mamba.cpu().tolist()
    case["phase"] = "compare_actual_device_bytes"
    for name, kind, tensor in buffers:
        raw = tensor.view(torch.uint8).reshape(tensor.shape[0], -1)
        actual = raw.index_select(0, destination_indices[kind]).cpu()
        wanted = expected[name]
        mismatch = (actual != wanted).flatten()
        bad = torch.nonzero(mismatch, as_tuple=False).flatten()
        first = int(bad[0]) if bad.numel() else None
        case["states"][name].update(
            status="pass" if first is None else "fail",
            actual_sha256=digest(actual),
            equal_bytes=int(actual.numel() - bad.numel()),
            mismatch_bytes=int(bad.numel()),
            poison_bytes_remaining=int((actual == 255).sum()),
            first_mismatch=None if first is None else {
                "byte_offset": first,
                "expected": int(wanted.flatten()[first]),
                "actual": int(actual.flatten()[first]),
            },
        )
    case["status"] = "pass" if all(row["status"] == "pass" for row in case["states"].values()) else "fail"
    case["phase"] = "release_host_pools"
    group.destroy()
    case["phase"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembler-override", type=Path, help="Explicit candidate assembler .py; default imports installed native module")
    parser.add_argument("--include-scaled-mla", action="store_true", help="Also exercise 656-byte scaled-FP8 MLA rows after the official 576-byte raw-FP8 layout")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 1 and 600")
    output_stream = sys.stdout
    started = time.monotonic()
    result = {
        "schema": 1,
        "status": "error",
        "backend": "direct",
        "host_layout": "page_first_direct",
        "assembler_override": str(args.assembler_override) if args.assembler_override else None,
        "coverage_limits": [
            "Single CUDA device, two full and two KDA layers, two pages and two state slots; no distributed rank coordination.",
            "Checks all stored bytes including index-K quantization scales, MLA payload/scales and KDA temporal/conv state; does not execute model projections or inference.",
            "Native L1/L2 only; no model weights, storage backend, dfkv/network traffic, MTP, speculative state or compressed index-kpool tails.",
        ],
        "cases": [],
    }
    done = threading.Event()

    def watchdog():
        if not done.wait(args.timeout_seconds):
            result.update(status="error", error="watchdog_timeout", elapsed_seconds=time.monotonic() - started)
            payload = (json.dumps(result, sort_keys=True) + "\n").encode()
            os.write(output_stream.fileno(), payload)
            os._exit(124)

    threading.Thread(target=watchdog, daemon=True).start()
    code = 2
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            import sglang
            from sglang.srt.runtime_context import get_context, get_parallel
            from sglang.srt.server_args import ServerArgs

            require(torch.cuda.is_available() and torch.version.cuda is not None, "A real NVIDIA CUDA runtime/device is required")
            require(os.environ.get("SGLANG_MOONCAKE_CUSTOM_MEM_POOL") is None, "Custom Mooncake allocator is outside this bounded no-network smoke")
            torch.cuda.set_device(0)
            result["runtime"] = {
                "sglang_version": getattr(sglang, "__version__", None),
                "sglang_path": sglang.__file__,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
            }
            server_args = ServerArgs(
                model_path="unused-no-model-weights",
                device="cuda",
                hicache_ratio=2.0,
                hicache_size=0,
                hicache_mem_layout="page_first_direct",
                hicache_io_backend="direct",
                hicache_storage_backend=None,
                hicache_write_policy="write_through",
                hicache_host_memory_mode="cache",
            )
            # Config projection only: unlike publish(), this does not resolve a
            # model config or load/download weights. Parallel overrides below
            # supply the only topology metadata needed by this single-rank path.
            get_context().set_server_args(server_args)
            if args.assembler_override:
                path = args.assembler_override.resolve(strict=True)
                spec = importlib.util.spec_from_file_location("sglang_hybrid_assembler_override", path)
                require(spec is not None and spec.loader is not None, "Cannot load assembler override")
                assembler = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = assembler
                spec.loader.exec_module(assembler)
            else:
                from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
            result["assembler_path"] = str(Path(assembler.__file__).resolve())
            result["assembler_sha256"] = hashlib.sha256(Path(assembler.__file__).read_bytes()).hexdigest()
            with get_parallel().override(
                tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
                attn_tp_rank=0, attn_tp_size=1, attn_cp_rank=0, attn_cp_size=1,
                attn_dp_rank=0, attn_dp_size=1, dcp_enabled=False,
                attn_dcp_rank=0, attn_dcp_size=1,
            ):
                layouts = ["raw-fp8"] + (["scaled-mla"] if args.include_scaled_mla else [])
                for layout in layouts:
                    case = {"layout": layout, "status": "error", "states": {}}
                    result["cases"].append(case)
                    try:
                        run_case(torch, assembler, case, server_args)
                    except Exception as exc:
                        case.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                        # A failed native CUDA operation can leave the context
                        # poisoned. Do not continue into another layout or pass.
                        break
            statuses = [case["status"] for case in result["cases"]]
            result["status"] = "error" if "error" in statuses else "fail" if "fail" in statuses else "pass"
            code = {"pass": 0, "fail": 1, "error": 2}[result["status"]]
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        done.set()
        print(json.dumps(result, sort_keys=True), file=output_stream, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
