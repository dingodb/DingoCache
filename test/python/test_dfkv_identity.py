"""Observable contract for connector-wide model and cache-object identity."""

import ctypes
import struct
import unittest

from dfkv_common import (
    DFKV_CLIENT_ABI_VERSION_V2,
    DFKV_CLIENT_OPT_REGISTER_WITH_MDS,
    DfkvClientOptionsV2,
    NamespaceDescriptor,
    ObjectDescriptor,
    PoolKind,
    RankCoordinate,
    canonical_namespace,
    layout_fingerprint,
    namespace_tenant_hash,
    reject_namespace_override,
    make_client_options_v2,
    make_key_array,
    make_key_buffer,
    pool_key,
    sg_key,
    tenant_hash,
)


class ClientOptionsV2Test(unittest.TestCase):
    def test_ctypes_layout_and_binary_namespace(self):
        self.assertEqual(ctypes.sizeof(DfkvClientOptionsV2), 88)
        namespace = b"model\x00layout"
        options = make_client_options_v2(
            namespace,
            members="n1=127.0.0.1:1",
            batch_concurrency=8,
            mds_endpoints="127.0.0.1:2",
            mds_group="g",
            register_client=True,
            client_id="worker-0",
            client_info="type=test",
        )
        self.assertEqual(
            options.struct_size, ctypes.sizeof(DfkvClientOptionsV2))
        self.assertEqual(options.abi_version, DFKV_CLIENT_ABI_VERSION_V2)
        self.assertEqual(
            options.flags, DFKV_CLIENT_OPT_REGISTER_WITH_MDS)
        self.assertEqual(options.batch_concurrency, 8)
        self.assertEqual(
            ctypes.string_at(options.key_namespace, options.key_namespace_len),
            namespace,
        )

    def test_binary_key_helpers_preserve_exact_pointer_lengths(self):
        keys = [b"a\x00\xff", b"a"]
        key_ptr, owner = make_key_buffer(keys[0])
        self.assertEqual(ctypes.string_at(key_ptr, len(keys[0])), keys[0])
        self.assertIsNotNone(owner)

        pointers, lengths, owners = make_key_array(keys)
        self.assertEqual(list(lengths), [3, 1])
        self.assertEqual(
            [ctypes.string_at(pointers[i], lengths[i]) for i in range(2)],
            keys,
        )
        self.assertEqual(len(owners), 2)


class CanonicalIdentityTest(unittest.TestCase):
    def test_namespace_and_object_match_cpp_golden_vectors(self):
        namespace = NamespaceDescriptor(
            tenant_id="tenant/a|b",
            model_id="m\x00x",
            model_revision="rev:42",
            pool_kind=PoolKind.MAMBA,
            pool_name="mamba.temporal",
            dtype="bf16",
            layout_fingerprint=0x1122334455667788,
            block_tokens=64,
            group_id=7,
            layer_begin=10,
            layer_count=22,
            tp=RankCoordinate(8, 3),
            dp=RankCoordinate(2, -1),
            pp=RankCoordinate(4, 1),
        )
        obj = ObjectDescriptor(
            logical_key=b"abc\x00def",
            component="conv:1",
            chunk_index=2,
            chunk_count=4,
        )
        self.assertEqual(
            namespace.serialize().hex(),
            "44464b564e5300020a00000074656e616e742f617c62030000006d0078"
            "060000007265763a343203000e0000006d616d62612e74656d706f72616c"
            "0400000062663136887766554433221140000000070000000a00000016000000"
            "080000000300000002000000ffffffff0400000001000000",
        )
        self.assertEqual(
            obj.serialize(namespace).hex(),
            "44464b564f424a327300000044464b564e5300020a00000074656e616e742f617c"
            "62030000006d0078060000007265763a343203000e0000006d616d62612e74656d"
            "706f72616c0400000062663136887766554433221140000000070000000a000000"
            "16000000080000000300000002000000ffffffff04000000010000000700000061"
            "62630064656606000000636f6e763a310200000004000000",
        )

    def test_automatic_namespace_uses_descriptor_and_layout_geometry(self):
        fields = {"page_size": 64, "is_mla": True, "dtype_tag": 7}
        actual = canonical_namespace(
            "org/GLM-5.2",
            "vllm/raw-v1",
            tenant_id="tenant-a",
            model_revision="rev-2026-08-02",
            dtype="fp8",
            block_tokens=64,
            layer_count=78,
            tp_size=8,
            dp_size=2,
            pp_size=1,
            layout_fields=fields,
        )
        expected = NamespaceDescriptor(
            tenant_id="tenant-a",
            model_id="org/GLM-5.2",
            model_revision="rev-2026-08-02",
            pool_kind=PoolKind.CUSTOM,
            pool_name="vllm/raw-v1",
            dtype="fp8",
            layout_fingerprint=layout_fingerprint("vllm/raw-v1", fields),
            block_tokens=64,
            group_id=0,
            layer_begin=0,
            layer_count=78,
            tp=RankCoordinate(8, -1),
            dp=RankCoordinate(2, -1),
            pp=RankCoordinate(1, -1),
        ).serialize()
        self.assertEqual(actual, expected)

    def test_tenant_hash_matches_cpp_vectors_and_namespace_fallback(self):
        self.assertEqual(tenant_hash("tenant-a"), 0x6164AF84ACBC9ADE)
        self.assertEqual(tenant_hash("tenant-b"), 0x27A61068BF52563B)
        namespace = canonical_namespace(
            "model", "vllm/raw-v1", tenant_id="tenant-a")
        self.assertEqual(
            namespace_tenant_hash(namespace), tenant_hash("tenant-a"))
        malformed = namespace + b"x"
        self.assertEqual(
            namespace_tenant_hash(malformed), tenant_hash(malformed))
        self.assertNotEqual(
            namespace_tenant_hash(malformed), tenant_hash("tenant-a"))
        semantic_malformed = bytearray(namespace)
        semantic_malformed[-4] = 1  # corrupt replicated-rank sentinel
        semantic_malformed = bytes(semantic_malformed)
        self.assertEqual(
            namespace_tenant_hash(semantic_malformed),
            tenant_hash(semantic_malformed),
        )

    def test_layout_fields_are_order_independent_and_identity_bearing(self):
        left = canonical_namespace(
            "model", "vllm/raw-v1",
            layout_fields={"dtype": "fp8", "page": 64})
        right = canonical_namespace(
            "model", "vllm/raw-v1",
            layout_fields={"page": 64, "dtype": "fp8"})
        changed = canonical_namespace(
            "model", "vllm/raw-v1",
            layout_fields={"page": 32, "dtype": "fp8"})
        self.assertEqual(left, right)
        self.assertNotEqual(left, changed)

    def test_default_namespaces_isolate_incompatible_raw_layouts(self):
        self.assertNotEqual(
            canonical_namespace("same-model", "vllm/raw-v1"),
            canonical_namespace("same-model", "sglang-hicache/raw-v1"),
        )

    def test_operator_namespace_override_is_rejected(self):
        with self.assertRaisesRegex(
                ValueError, "namespace identity is derived automatically"):
            reject_namespace_override(
                {"key_namespace": "org/model@layout=unreviewed"})

        with self.assertRaisesRegex(
                ValueError, "remote_storage_plugin.dfkv.key_namespace"):
            reject_namespace_override(
                {"remote_storage_plugin.dfkv.key_namespace": ""},
                key="remote_storage_plugin.dfkv.key_namespace",
            )

    def test_pool_key_matches_binary_length_framed_contract(self):
        page_hash = b"\xffhash\x00|="
        key = pool_key(
            page_hash,
            pool=b"mamba\x00pool",
            dp_size=2,
            dp_rank=-1,
            tp_size=8,
            tp_rank=3,
            pcp_size=2,
            pcp_rank=1,
            dcp_size=4,
            dcp_rank=2,
            pp_size=3,
            pp_rank=1,
            group_id=7,
            component=b"temporal\x00|=",
        )
        self.assertEqual(
            key,
            b"DFKVPOOL\x02"
            + struct.pack("<I", 10) + b"mamba\x00pool"
            + struct.pack("<I", len(page_hash)) + page_hash
            + struct.pack(
                "<IiIiIiIiIiI",
                2, -1, 8, 3, 2, 1, 4, 2, 3, 1, 7,
            )
            + struct.pack("<I", 11) + b"temporal\x00|=",
        )

    def test_layout_or_object_change_never_aliases(self):
        base = dict(pool="kv", tp_size=8, tp_rank=0)
        self.assertNotEqual(pool_key("page-a", **base), pool_key("page-b", **base))
        self.assertNotEqual(
            pool_key("page-a", **base),
            pool_key("page-a", pool="kv", tp_size=4, tp_rank=0),
        )
        self.assertNotEqual(
            pool_key("page-a", **base),
            pool_key("page-a", pool="kv", tp_size=8, tp_rank=1),
        )

    def test_pool_key_binary_fields_cannot_alias(self):
        self.assertNotEqual(pool_key(b"a\x00b"), pool_key(b"a"))
        self.assertNotEqual(
            pool_key(b"a", pool=b"x\x00y"),
            pool_key(b"a", pool=b"x"),
        )
        self.assertNotEqual(
            pool_key(b"a", component=b"\xff\x00"),
            pool_key(b"a", component=b"\xff"),
        )

    def test_invalid_coordinates_fail_fast(self):
        for kwargs in (
            {"tp_size": 0, "tp_rank": 0},
            {"tp_size": 8, "tp_rank": 8},
            {"tp_size": 8, "tp_rank": -2},
        ):
            with self.assertRaises(ValueError):
                pool_key("page", **kwargs)

    def test_scatter_groups_are_explicit_binary_coordinates(self):
        logical = pool_key(b"page\x00\xff", tp_size=1, tp_rank=0)
        self.assertEqual(
            sg_key(logical, 29, 2),
            logical + b"DFKVSG\x02" + struct.pack("<II", 29, 2),
        )


if __name__ == "__main__":
    unittest.main()
