"""Packed-GQA FP8 paged decode/verify kernel used for Gemma 4 attention on SM120.

Checks the Triton kernel against an FP32 softmax reference on small paged caches:
ragged query lengths with bottom-right causal masking, KV splits that leave some
splits empty, the head_dim=256 sliding-window path, and the contract gate that keeps
unsupported calls on FA4.
"""

import unittest

import torch

from sglang.kernels.ops.attention.paged_gqa_fp8_sm120 import (
    can_use_paged_gqa_fp8,
    paged_gqa_fp8_attention,
    short_query_tail,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

FP8 = torch.float8_e4m3fn
NUM_Q_HEADS = 32
K_SCALE, V_SCALE = 0.75, 1.25


def _reference(q, k, v, page_table, seqlens, q_lens, softmax_scale, window):
    group = q.shape[1] // k.shape[2]
    outs, start = [], 0
    for b, (n, q_len) in enumerate(zip(seqlens.tolist(), q_lens)):
        slots = page_table[b, :n].long()
        keys = (k[slots, 0].float() * K_SCALE).repeat_interleave(group, 1)
        values = (v[slots, 0].float() * V_SCALE).repeat_interleave(group, 1)
        scores = torch.einsum("thd,nhd->htn", q[start : start + q_len].float(), keys)
        visible_end = n - q_len + torch.arange(q_len, device=q.device)
        pos = torch.arange(n, device=q.device)[None, :]
        hidden = pos > visible_end[:, None]
        if window >= 0:
            hidden |= pos < visible_end[:, None] - window
        scores = (scores * softmax_scale).masked_fill(hidden[None], float("-inf"))
        # Stale NaN slots get zero weight; zero their values so 0 * NaN stays 0.
        probs = scores.softmax(-1)
        values = torch.where(probs.sum((0, 1))[:, None, None] > 0, values, 0.0)
        outs.append(torch.einsum("htn,nhd->thd", probs, values))
        start += q_len
    return torch.cat(outs)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires SM120",
)
class TestPagedGqaFp8Sm120(CustomTestCase):
    def _run(self, seqlens, q_lens, num_splits, num_kv_heads=4, head_dim=512, window=-1,
             shared=None, cascade=None, short_tail=None):
        torch.manual_seed(0)
        dev = "cuda"
        num_slots = sum(seqlens) + 64
        k = (torch.randn(num_slots, 1, num_kv_heads, head_dim, device=dev) * 0.5).to(FP8)
        v = (torch.randn(num_slots, 1, num_kv_heads, head_dim, device=dev) * 0.5).to(FP8)
        # Scatter each request over random, non-contiguous slots; the last slot is
        # reserved for NaN garbage.
        perm = torch.randperm(num_slots - 1, device=dev).int()
        stale = num_slots - 1
        k[stale] = float("nan")
        v[stale] = float("nan")
        page_table = torch.zeros(len(seqlens), max(seqlens) + 8, dtype=torch.int32, device=dev)
        offset = 0
        for b, (n, q_len) in enumerate(zip(seqlens, q_lens)):
            page_table[b, :n] = perm[offset : offset + n]
            offset += n
            if shared is not None and b > 0:
                # Request b reuses request 0's first shared[b] slots (a cached prompt).
                page_table[b, : shared[b]] = page_table[0, : shared[b]]
            if window >= 0:
                # Keys before the earliest query's window must never be read.
                page_table[b, : max(n - q_len - window, 0)] = stale
        cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()],
                                    dtype=torch.int32, device=dev)
        # Large query norm gives peaked attention, so a wrong mask or head shows up.
        q = (torch.randn(sum(q_lens), NUM_Q_HEADS, head_dim, device=dev) * 3).to(FP8)
        seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
        softmax_scale = head_dim**-0.5
        out = paged_gqa_fp8_attention(
            q=q, key_cache=k, value_cache=v, page_table=page_table,
            cache_seqlens=seqlens_t, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max(q_lens), softmax_scale=softmax_scale,
            k_scale=torch.tensor(K_SCALE, device=dev),
            v_scale=torch.tensor(V_SCALE, device=dev), window_left=window,
            num_splits=num_splits, cascade=cascade, short_tail=short_tail,
        )
        ref = _reference(q, k, v, page_table, seqlens_t, q_lens, softmax_scale, window)
        self.assertEqual(out.dtype, torch.bfloat16)
        err = (out.float() - ref).abs().max().item()
        self.assertLess(err, 0.04 * ref.abs().max().item())

    def test_verify_single_split(self):
        # cascade=False: the only path storing straight to the output without a combine.
        self._run(seqlens=[37, 300, 129], q_lens=[4, 4, 4], num_splits=1, cascade=False)

    def test_ragged_queries_with_empty_splits(self):
        # 37 tokens over 4 splits of 32 leaves splits 2 and 3 empty.
        self._run(seqlens=[37, 700, 5], q_lens=[4, 1, 3], num_splits=4)

    def test_decode_many_splits(self):
        self._run(seqlens=[1000, 64], q_lens=[1, 1], num_splits=16)

    def test_prefill_ragged_with_prefix(self):
        # 100 x 8 packed rows exceeds the short-query tile, so the whole batch takes the
        # prefill kernel: a chunk after a cached prefix, a prefix-free prompt, and a
        # 9-token chunk whose last M tile is mostly padding.
        self._run(seqlens=[300, 37, 129], q_lens=[100, 37, 9], num_splits=None)

    def test_mixed_chunk_short_tail(self):
        # A mixed chunked-prefill batch: a 100-token chunk and a 37-token prompt, then 17
        # one-token decode rows sharing a 600-key prompt (enough rows for the cascade).
        q_lens = [100, 37] + [1] * 17
        tail = short_query_tail(q_lens, group=8)
        self.assertEqual(tail, (2, 137, 1))
        self._run(seqlens=[300, 37] + [700 + 3 * i for i in range(17)], q_lens=q_lens,
                  num_splits=None, shared=[0, 0] + [600] * 17, short_tail=tail)

    def test_short_query_tail_split_point(self):
        # 8 x 8 rows still fits the packed tile, 9 x 8 does not.
        self.assertEqual(short_query_tail([300, 9, 8, 1, 4], group=8), (2, 309, 8))
        # A long last request, an all-short batch, and an empty-only tail give no split.
        self.assertIsNone(short_query_tail([1, 300], group=8))
        self.assertIsNone(short_query_tail([4, 1, 2], group=8))
        self.assertIsNone(short_query_tail([300, 0], group=8))

    def test_cascade_shared_prefix(self):
        # Requests 0, 1 and 3 share a 600-key prompt (not a multiple of the 64-key block),
        # request 2 shares only 30 keys and the last row mimics a CUDA-graph padding
        # request; the shared prefix must still be picked and the others served alone.
        self._run(seqlens=[700, 650, 40, 690, 1], q_lens=[4, 4, 4, 4, 1], num_splits=None,
                  shared=[0, 600, 30, 600, 0], cascade=True)

    def test_sliding_window_head_dim_256(self):
        # Window 100 over 4 splits: request 0 starts its window at key 296, request 1
        # is shorter than the window, request 2 sits exactly at the window edge.
        self._run(seqlens=[400, 60, 104], q_lens=[4, 2, 4], num_splits=4,
                  num_kv_heads=16, head_dim=256, window=100)

    def test_gate(self):
        q = torch.empty(4, NUM_Q_HEADS, 512, dtype=FP8)
        cache = torch.empty(8, 1, 4, 512, dtype=FP8)
        scale = torch.ones(())
        ok = dict(q=q, key_cache=cache, max_seqlen_q=4, window_size=(-1, -1),
                  softcap=0.0, causal=True, k_scale=scale, v_scale=scale)
        self.assertTrue(can_use_paged_gqa_fp8(**ok))
        swa = {**ok, "window_size": (1023, 0), "q": torch.empty(4, 32, 256, dtype=FP8),
               "key_cache": torch.empty(8, 1, 16, 256, dtype=FP8)}
        self.assertTrue(can_use_paged_gqa_fp8(**swa))
        # 9 x 8 packed rows exceeds the short-query tile; full HD512 takes the prefill path.
        self.assertTrue(can_use_paged_gqa_fp8(**{**ok, "max_seqlen_q": 9}))
        # 33 x 2 packed rows exceeds the short-query tile; windows have no prefill path.
        self.assertFalse(can_use_paged_gqa_fp8(**{**swa, "max_seqlen_q": 33}))
        for change in (
            {"window_size": (1023, 1023)},
            {"window_size": (-1, 0)},
            {"softcap": 30.0},
            {"causal": False},
            {"k_scale": None},
            {"q": q.to(torch.bfloat16)},
            {"key_cache": torch.empty(8, 16, 4, 512, dtype=FP8)},
            {"key_cache": torch.empty(8, 1, 4, 128, dtype=FP8),
             "q": torch.empty(4, 32, 128, dtype=FP8)},
        ):
            with self.subTest(change=list(change)):
                self.assertFalse(can_use_paged_gqa_fp8(**{**ok, **change}))


if __name__ == "__main__":
    unittest.main()
