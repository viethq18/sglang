"""SGLANG_FA4_SM90_FP8_PREFILL_VIA_FA3: FP8 prefill routed from FA4 to FA3 on SM90
must match an fp32 reference and return FA4's output/LSE layout."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.flash_attention_v3 import (
    flash_attn_with_kvcache as fa3_flash_attn_with_kvcache,
)
from sglang.kernels.ops.attention.flash_attention_v4 import (
    flash_attn_with_kvcache as fa4_flash_attn_with_kvcache,
)
from sglang.srt.layers.attention.flashattention_backend import (
    _route_fp8_prefill_to_fa3,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

skip_condition = not torch.cuda.is_available() or (
    torch.cuda.get_device_capability()[0] != 9
)

H, HK, D = 24, 4, 256
FP8 = torch.float8_e4m3fn


def _reference(q, k_cache, v_cache, page_table, prefix_lens, q_lens, k_scale, v_scale):
    outs, lses = [], []
    start = 0
    for b, (p, t) in enumerate(zip(prefix_lens, q_lens)):
        slots = page_table[b, : p + t].long()
        k = k_cache[slots, 0].float() * k_scale
        v = v_cache[slots, 0].float() * v_scale
        k = k.repeat_interleave(H // HK, dim=1)
        v = v.repeat_interleave(H // HK, dim=1)
        qb = q[start : start + t].float()
        s = torch.einsum("qhd,khd->hqk", qb, k) * D**-0.5
        q_pos = torch.arange(t, device=q.device)[:, None] + p
        k_pos = torch.arange(p + t, device=q.device)[None, :]
        s = s.masked_fill(k_pos > q_pos, float("-inf"))
        lses.append(torch.logsumexp(s, dim=-1))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), v))
        start += t
    return torch.cat(outs), torch.cat(lses, dim=1)


@pytest.mark.skipif(skip_condition, reason="FA3 FP8 routing is SM90 only.")
@pytest.mark.parametrize("q_lens,prefix_lens", [((512,), (0,)), ((300, 1024, 64), (700, 0, 2048))])
def test_routed_fp8_prefill_matches_reference_and_fa4_layout(q_lens, prefix_lens):
    torch.manual_seed(0)
    dev = "cuda"
    batch = len(q_lens)
    total_k = [p + t for p, t in zip(prefix_lens, q_lens)]
    slots = 4 * sum(total_k)
    k_scale, v_scale = 0.5, 2.0
    k_cache = (torch.randn(slots, 1, HK, D, device=dev)).to(FP8)
    v_cache = (torch.randn(slots, 1, HK, D, device=dev) * 0.25).to(FP8)
    page_table = torch.zeros(batch, max(total_k), dtype=torch.int32, device=dev)
    perm = torch.randperm(slots, device=dev).to(torch.int32)
    offset = 0
    for b, n in enumerate(total_k):
        page_table[b, :n] = perm[offset : offset + n]
        offset += n
    q = (torch.randn(sum(q_lens), H, D, device=dev) * 0.5).to(FP8)
    cu_q = torch.tensor([0, *q_lens], device=dev, dtype=torch.int32).cumsum(0).int()
    kwargs = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor(total_k, device=dev, dtype=torch.int32),
        cu_seqlens_q=cu_q,
        cu_seqlens_k_new=None,
        max_seqlen_q=max(q_lens),
        softmax_scale=D**-0.5,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        return_softmax_lse=True,
        num_splits=0,
        k_descale=torch.full((batch, HK), k_scale, device=dev),
        v_descale=torch.full((batch, HK), v_scale, device=dev),
        score_mod=None,
        aux_tensors=None,
        rel_bias=None,
        rel_bias_prep_cache=None,
    )
    routed = _route_fp8_prefill_to_fa3(
        fa4_flash_attn_with_kvcache, fa3_flash_attn_with_kvcache
    )
    out, lse, *_ = routed(**kwargs)
    out_fa4, lse_fa4, *_ = fa4_flash_attn_with_kvcache(**kwargs)
    ref_out, ref_lse = _reference(
        q, k_cache, v_cache, page_table, prefix_lens, q_lens, k_scale, v_scale
    )

    assert out.shape == out_fa4.shape and out.dtype == out_fa4.dtype
    assert lse.shape == lse_fa4.shape
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()  # noqa: E731
    # FA3 and FA4 both quantize P to E4M3; FA3 must be no less accurate.
    assert rel(out, ref_out) < max(2e-2, 1.5 * rel(out_fa4, ref_out))
    torch.testing.assert_close(lse.float(), ref_lse, atol=2e-2, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
