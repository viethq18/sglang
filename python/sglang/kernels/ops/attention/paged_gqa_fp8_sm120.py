# SPDX-License-Identifier: Apache-2.0
"""Paged GQA attention over an FP8 E4M3 KV cache, in Triton, for SM120.

Serves Gemma 4's attention layers where the FA4 SM120 kernel is weakest:

* Short queries (decode, and target verify / draft decode of speculative decoding) of
  the head_dim=512 full-attention layers and the head_dim=256 sliding-window layers.
  FA4 tiles one query head group per CTA and re-reads the KV once per 32-row Q tile.
  Here one program owns one (request, KV head, KV split) and packs all q_len x
  group_size query rows of that request that share the KV head into one M tile, so
  each KV token is loaded from HBM once per KV head. With more than one split, every
  split writes an FP32 partial output and a base-2 LSE, and a combine kernel merges
  them.
* Long queries (prefill, chunked prefill) of the head_dim=512 full-attention layers.
  FA4's SMEM budget limits HD512 to 32x32 tiles; here one program owns a 64-row packed
  tile of one (request, KV head) and streams 64-key blocks, with the full HD512 V
  row in registers. Keys below the tile's first query position skip the causal mask.

QK and PV run as E4M3 tensor-core dots with FP32 accumulation; the probabilities are
scaled by 256 before the E4M3 cast so small weights keep precision, and the scale is
removed in the epilogue.

Contract (all checked by `can_use_paged_gqa_fp8`):
  q            [num_tokens, num_q_heads, D]   E4M3, rows of request b at cu_seqlens_q[b]
  k/v cache    [num_slots, 1, num_kv_heads, D] E4M3 (page size one)
  page_table   [batch, >= max cache_seqlens]  int32 slot ids
  cache_seqlens[batch]                          int32, KV length including the new tokens
  Causal masking is bottom-right aligned: query t of a q_len-row request sees keys
  n <= cache_seqlens - q_len + t, and with a left window W also n >= that bound - W.
  Sliding windows are served for short queries only. Keys before the earliest window
  are never read, so their page table entries may be stale. No softcap, sinks or
  score mods.
"""

from collections.abc import Sequence
from typing import Optional

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_SPLIT_CHOICES = (1, 2, 4, 8, 16)
# Packed M tile (q_len x group_size rows) upper bound for the short-query kernel; larger
# tiles spill registers, so longer queries take the prefill kernel.
MAX_PACKED_ROWS = 64
_PREFILL_BM, _PREFILL_BN = 64, 64
_PREFIX_SPLITS = 4
_CASCADE_MIN_BATCH = 16


@triton.jit
def _paged_gqa_fwd(
    Q, K, V, PT, SL, CUQ, KS, VS, O, OP, LSEP, NLO,
    qk_scale_log2,
    s_qt, s_qh, s_kt, s_kh, s_vt, s_vh, s_pt, s_ot, s_oh,
    s_opt, s_oph, s_ops, s_lt, s_lh,
    G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    SPLITS: tl.constexpr, WINDOW: tl.constexpr, CASCADE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    q_start = tl.load(CUQ + b)
    q_len = tl.load(CUQ + b + 1) - q_start
    seqlen = tl.load(SL + b)

    rows = tl.arange(0, BM)
    t = rows // G
    head = h * G + rows % G
    row_ok = t < q_len
    d = tl.arange(0, D)
    tok = (q_start + t).to(tl.int64)

    n_lo = 0
    if WINDOW >= 0:
        n_lo = tl.maximum(seqlen - q_len - WINDOW, 0)
    if CASCADE:
        # Keys below NLO[b] belong to the shared prefix, served by _paged_gqa_prefix.
        n_lo = tl.load(NLO + b)
    chunk = tl.cdiv(tl.cdiv(seqlen - n_lo, SPLITS), BN) * BN
    n_start = n_lo + sp * chunk
    n_end = tl.minimum(n_start + chunk, seqlen)

    q = tl.load(Q + tok[:, None] * s_qt + head[:, None] * s_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    qk_scale = qk_scale_log2 * tl.load(KS).to(tl.float32)
    qpos = seqlen - q_len + t

    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(n_start, n_end, BN):
        n = n0 + tl.arange(0, BN)
        n_ok = n < n_end
        slot = tl.load(PT + b * s_pt + n, mask=n_ok, other=0).to(tl.int64)
        k = tl.load(K + slot[:, None] * s_kt + h * s_kh + d[None, :],
                    mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        visible = n_ok[None, :] & (n[None, :] <= qpos[:, None])
        if WINDOW >= 0:
            visible = visible & (n[None, :] >= qpos[:, None] - WINDOW)
        s = tl.where(visible, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + slot[:, None] * s_vt + h * s_vh + d[None, :],
                    mask=n_ok[:, None], other=0.0)
        acc = tl.dot((p * 256.0).to(tl.float8e4nv), v, acc * alpha[:, None])
        m_i = m_new

    pv_scale = tl.load(VS).to(tl.float32) * (1.0 / 256.0)
    out = acc * (pv_scale / tl.where(l_i == 0.0, 1.0, l_i))[:, None]
    if SPLITS == 1 and not CASCADE:
        tl.store(O + tok[:, None] * s_ot + head[:, None] * s_oh + d[None, :],
                 out.to(O.dtype.element_ty), mask=row_ok[:, None])
    else:
        lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log2(l_i))
        tl.store(OP + tok[:, None] * s_opt + head[:, None] * s_oph + sp * s_ops + d[None, :],
                 out, mask=row_ok[:, None])
        tl.store(LSEP + tok * s_lt + head * s_lh + sp, lse, mask=row_ok)


@triton.jit
def _shared_prefix_len(PT, SL, CUQ, LCP, s_pt, BLK: tl.constexpr):
    """LCP[b] = leading page-table slots request b shares with request 0, capped at the
    first query position of b so every query row of b sees the whole prefix."""
    b = tl.program_id(0)
    limit = tl.load(SL + b) - (tl.load(CUQ + b + 1) - tl.load(CUQ + b))
    lcp = tl.maximum(limit, 0)
    n0 = 0
    while n0 < lcp:
        n = n0 + tl.arange(0, BLK)
        ok = n < lcp
        mine = tl.load(PT + b * s_pt + n, mask=ok, other=0)
        ref = tl.load(PT + n, mask=ok, other=0)
        lcp = tl.minimum(lcp, tl.min(tl.where(ok & (mine != ref), n, lcp), 0))
        n0 += BLK
    tl.store(LCP + b, lcp)


@triton.jit
def _select_shared_prefix(LCP, NLO, batch, BP2: tl.constexpr):
    """Pick the prefix length P maximising P x #requests sharing it (graph padding rows and
    requests outside the shared prompt do not drag P to zero). NLO[b] = P for requests that
    share it, else 0; NLO[batch] = P."""
    i = tl.arange(0, BP2)
    ok = i < batch
    lcp = tl.load(LCP + i, mask=ok, other=0)
    shares = (lcp[:, None] >= lcp[None, :]) & ok[:, None]
    score = tl.where(ok, tl.sum(shares.to(tl.int64), 0) * lcp.to(tl.int64), -1)
    p = tl.max(tl.where(score == tl.max(score, 0), lcp, 0), 0)
    tl.store(NLO + i, tl.where(lcp >= p, p, 0), mask=ok)
    tl.store(NLO + batch, p)


@triton.jit
def _paged_gqa_prefix(
    Q, K, V, PT, NLO, KS, VS, OP, LSEP,
    qk_scale_log2, num_rows, batch,
    s_qt, s_qh, s_kt, s_kh, s_vt, s_vh,
    s_opt, s_oph, s_ops, s_lt, s_lh,
    G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    PSPLITS: tl.constexpr, SLOT0: tl.constexpr,
):
    """Shared-prefix part of a cascade: one program owns 64 packed (token, q head) rows of the
    whole batch for one KV head and one split of keys [0, P), read via request 0's pages.
    Every row sees every prefix key, so no mask beyond the key bound."""
    mt = tl.program_id(0)
    h = tl.program_id(1)
    ps = tl.program_id(2)
    prefix = tl.load(NLO + batch)
    rows = mt * BM + tl.arange(0, BM)
    row_ok = rows < num_rows
    tok = (rows // G).to(tl.int64)
    head = h * G + rows % G
    d = tl.arange(0, D)
    chunk = tl.cdiv(tl.cdiv(prefix, PSPLITS), BN) * BN
    n_start = ps * chunk
    n_end = tl.minimum(n_start + chunk, prefix)

    q = tl.load(Q + tok[:, None] * s_qt + head[:, None] * s_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    qk_scale = qk_scale_log2 * tl.load(KS).to(tl.float32)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(n_start, n_end, BN):
        n = n0 + tl.arange(0, BN)
        n_ok = n < n_end
        slot = tl.load(PT + n, mask=n_ok, other=0).to(tl.int64)
        k = tl.load(K + slot[:, None] * s_kt + h * s_kh + d[None, :],
                    mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(n_ok[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + slot[:, None] * s_vt + h * s_vh + d[None, :],
                    mask=n_ok[:, None], other=0.0)
        acc = tl.dot((p * 256.0).to(tl.float8e4nv), v, acc * alpha[:, None])
        m_i = m_new

    pv_scale = tl.load(VS).to(tl.float32) * (1.0 / 256.0)
    out = acc * (pv_scale / tl.where(l_i == 0.0, 1.0, l_i))[:, None]
    lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log2(l_i))
    slot_idx = SLOT0 + ps
    tl.store(OP + tok[:, None] * s_opt + head[:, None] * s_oph + slot_idx * s_ops + d[None, :],
             out, mask=row_ok[:, None])
    tl.store(LSEP + tok * s_lt + head * s_lh + slot_idx, lse, mask=row_ok)



@triton.jit
def _paged_gqa_prefill(
    Q, K, V, PT, SL, CUQ, KS, VS, O,
    qk_scale_log2,
    s_qt, s_qh, s_kt, s_kh, s_vt, s_vh, s_pt, s_ot, s_oh,
    G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    # Launch the last (longest causal) M tiles first so the tail wave is short.
    mb = tl.num_programs(0) - 1 - tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    q_start = tl.load(CUQ + b)
    q_len = tl.load(CUQ + b + 1) - q_start
    if mb * BM >= q_len * G:
        return
    seqlen = tl.load(SL + b)

    rows = mb * BM + tl.arange(0, BM)
    t = rows // G
    head = h * G + rows % G
    row_ok = t < q_len
    d = tl.arange(0, D)
    tok = (q_start + t).to(tl.int64)
    q = tl.load(Q + tok[:, None] * s_qt + head[:, None] * s_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    qk_scale = qk_scale_log2 * tl.load(KS).to(tl.float32)
    qpos = seqlen - q_len + t
    # Keys below the tile's first query position are visible to every row.
    full_end = ((seqlen - q_len + mb * BM // G + 1) // BN) * BN
    n_end = seqlen - q_len + tl.minimum(mb * BM + BM - 1, q_len * G - 1) // G + 1
    pt_row = PT + b * s_pt

    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(0, full_end, BN):
        slot = tl.load(pt_row + n0 + tl.arange(0, BN)).to(tl.int64)
        k = tl.load(K + slot[:, None] * s_kt + h * s_kh + d[None, :])
        s = tl.dot(q, tl.trans(k)) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + slot[:, None] * s_vt + h * s_vh + d[None, :])
        acc = tl.dot((p * 256.0).to(tl.float8e4nv), v, acc * alpha[:, None])
        m_i = m_new
    for n0 in range(full_end, n_end, BN):
        n = n0 + tl.arange(0, BN)
        n_ok = n < n_end
        slot = tl.load(pt_row + n, mask=n_ok, other=0).to(tl.int64)
        k = tl.load(K + slot[:, None] * s_kt + h * s_kh + d[None, :],
                    mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(n_ok[None, :] & (n[None, :] <= qpos[:, None]), s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + slot[:, None] * s_vt + h * s_vh + d[None, :],
                    mask=n_ok[:, None], other=0.0)
        acc = tl.dot((p * 256.0).to(tl.float8e4nv), v, acc * alpha[:, None])
        m_i = m_new

    pv_scale = tl.load(VS).to(tl.float32) * (1.0 / 256.0)
    out = acc * (pv_scale / tl.where(l_i == 0.0, 1.0, l_i))[:, None]
    tl.store(O + tok[:, None] * s_ot + head[:, None] * s_oh + d[None, :],
             out.to(O.dtype.element_ty), mask=row_ok[:, None])


@triton.jit
def _paged_gqa_combine(
    OP, LSEP, CUQ, NLO, O,
    s_opt, s_oph, s_ops, s_lt, s_lh, s_ot, s_oh,
    MAX_T: tl.constexpr, SPLITS: tl.constexpr, SP2: tl.constexpr, D: tl.constexpr,
    SUFFIX_SPLITS: tl.constexpr,
):
    """Merge SPLITS partial (output, base-2 LSE) slots per query row. With a cascade, slots
    from SUFFIX_SPLITS on hold the shared prefix and only count for requests sharing it."""
    b = tl.program_id(0) // MAX_T
    t = tl.program_id(0) % MAX_T
    hq = tl.program_id(1)
    q_start = tl.load(CUQ + b)
    if t < tl.load(CUQ + b + 1) - q_start:
        tok = (q_start + t).to(tl.int64)
        s = tl.arange(0, SP2)
        s_ok = s < SPLITS
        if SUFFIX_SPLITS < SPLITS:
            s_ok = s_ok & ((s < SUFFIX_SPLITS) | (tl.load(NLO + b) > 0))
        lse = tl.load(LSEP + tok * s_lt + hq * s_lh + s, mask=s_ok, other=-float("inf"))
        m = tl.max(lse, 0)
        w = tl.exp2(lse - tl.where(m == -float("inf"), 0.0, m))
        w = w / tl.maximum(tl.sum(w, 0), 1e-30)
        d = tl.arange(0, D)
        o = tl.load(OP + tok * s_opt + hq * s_oph + s[:, None] * s_ops + d[None, :],
                    mask=s_ok[:, None], other=0.0)
        tl.store(O + tok * s_ot + hq * s_oh + d,
                 tl.sum(o * w[:, None], 0).to(O.dtype.element_ty))


def can_use_paged_gqa_fp8(
    *,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    max_seqlen_q: int,
    window_size: tuple[int, int],
    softcap: float,
    causal: bool,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
) -> bool:
    """Whether a paged-KV attention call fits the kernel contract above."""
    if key_cache.dim() != 4 or key_cache.shape[1] != 1:
        return False
    num_kv_heads, head_dim = key_cache.shape[2], key_cache.shape[3]
    num_q_heads = q.shape[-2]
    group = num_q_heads // num_kv_heads
    short_query = max_seqlen_q * group <= MAX_PACKED_ROWS
    return (
        q.dtype == torch.float8_e4m3fn
        and key_cache.dtype == torch.float8_e4m3fn
        and head_dim in (256, 512)
        and num_q_heads == group * num_kv_heads
        and causal
        and (tuple(window_size) == (-1, -1) or (window_size[0] >= 0 and window_size[1] == 0))
        and not softcap
        and k_scale is not None
        and v_scale is not None
        and k_scale.numel() == 1
        and v_scale.numel() == 1
        and 1 <= max_seqlen_q
        and (short_query or (head_dim == 512 and tuple(window_size) == (-1, -1)))
    )


def select_num_splits(batch: int, num_kv_heads: int, num_sms: int) -> int:
    """Smallest KV split count that gives every SM about four programs."""
    for splits in _SPLIT_CHOICES:
        if batch * num_kv_heads * splits >= 4 * num_sms:
            return splits
    return _SPLIT_CHOICES[-1]


def short_query_tail(extend_seq_lens: Sequence[int], group: int) -> Optional[tuple[int, int, int]]:
    """Split point of a long-query batch whose trailing requests are short queries.

    Mixed chunked prefill appends the running decode rows (one token each) after the prefill
    requests. Returns (first_req, first_token, max_q) of the longest run of trailing requests
    with q_len x group <= MAX_PACKED_ROWS, or None when there is no such run or no request
    precedes it.
    """
    first_req = len(extend_seq_lens)
    max_q = 0
    while first_req > 0 and extend_seq_lens[first_req - 1] * group <= MAX_PACKED_ROWS:
        first_req -= 1
        max_q = max(max_q, extend_seq_lens[first_req])
    if first_req == 0 or first_req == len(extend_seq_lens) or max_q == 0:
        return None
    return first_req, sum(extend_seq_lens[:first_req]), max_q


def paged_gqa_fp8_attention(
    *,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    window_left: int = -1,
    out: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    cascade: Optional[bool] = None,
    short_tail: Optional[tuple[int, int, int]] = None,
) -> torch.Tensor:
    """Attention of `q` against the paged FP8 KV cache; returns BF16 [num_tokens, Hq, D].

    Short full-attention queries of batches of >= 16 requests run as a cascade: the longest page
    table prefix shared with request 0 (the common system prompt) is attended once per
    64 packed rows of the whole batch, and each request only streams its own suffix.

    `short_tail` (from `short_query_tail`) splits a long-query batch: the trailing short
    requests (mixed-chunk decode rows) take the short-query kernel instead of padding a
    64-row prefill tile each and streaming the whole KV per request.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    batch = cache_seqlens.shape[0]
    num_kv_heads = key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    k = key_cache.view(-1, num_kv_heads, head_dim)
    v = value_cache.view(-1, num_kv_heads, head_dim)
    if out is None:
        out = torch.empty(num_tokens, num_q_heads, head_dim, dtype=torch.bfloat16,
                          device=q.device)
    if max_seqlen_q * group > MAX_PACKED_ROWS and short_tail is not None:
        first_req, first_token, tail_max_q = short_tail
        common = dict(key_cache=key_cache, value_cache=value_cache, softmax_scale=softmax_scale,
                      k_scale=k_scale, v_scale=v_scale, window_left=window_left)
        paged_gqa_fp8_attention(
            q=q[:first_token], page_table=page_table[:first_req],
            cache_seqlens=cache_seqlens[:first_req], cu_seqlens_q=cu_seqlens_q[: first_req + 1],
            max_seqlen_q=max_seqlen_q, out=out[:first_token], **common,
        )
        paged_gqa_fp8_attention(
            q=q[first_token:], page_table=page_table[first_req:],
            cache_seqlens=cache_seqlens[first_req:],
            cu_seqlens_q=cu_seqlens_q[first_req:] - first_token,
            max_seqlen_q=tail_max_q, out=out[first_token:], num_splits=num_splits,
            cascade=cascade, **common,
        )
        return out
    if max_seqlen_q * group > MAX_PACKED_ROWS:
        grid = (triton.cdiv(max_seqlen_q * group, _PREFILL_BM), batch, num_kv_heads)
        _paged_gqa_prefill[grid](
            q, k, v, page_table, cache_seqlens, cu_seqlens_q, k_scale, v_scale, out,
            softmax_scale * _LOG2E,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            page_table.stride(0), out.stride(0), out.stride(1),
            G=group, D=head_dim, BM=_PREFILL_BM, BN=_PREFILL_BN, num_warps=8, num_stages=1,
        )
        return out
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    if cascade is None:
        # Below ~16 requests the prefix pass has too few programs to beat streaming the
        # (then mostly L2-resident) shared prefix per request.
        cascade = window_left < 0 and batch >= _CASCADE_MIN_BATCH
    if num_splits is None:
        num_splits = select_num_splits(batch, num_kv_heads, num_sms)
        if cascade:
            # Requests only stream their own suffix, so half the key splits suffice.
            num_splits = max(1, num_splits // 2)
    prefix_splits = 0
    if cascade:
        row_tiles = triton.cdiv(num_tokens * group, _PREFILL_BM)
        # Prefix programs are long (thousands of keys); 4 splits keep the wave tail small.
        prefix_splits = _PREFIX_SPLITS
    total_splits = num_splits + prefix_splits
    packed_rows = max(16, triton.next_power_of_2(max_seqlen_q * group))
    if total_splits > 1:
        partial = torch.empty(num_tokens, num_q_heads, total_splits, head_dim,
                              dtype=torch.float32, device=q.device)
        lse = torch.empty(num_tokens, num_q_heads, total_splits, dtype=torch.float32,
                          device=q.device)
    else:
        partial = out.new_empty(1, 1, 1, 1, dtype=torch.float32)
        lse = out.new_empty(1, 1, 1, dtype=torch.float32)
    # nlo[:batch]: first key each request streams itself; nlo[batch]: shared prefix length.
    nlo = torch.empty(2 * batch + 1 if cascade else 1, dtype=torch.int32, device=q.device)
    if cascade:
        lcp = nlo[batch + 1:]
        _shared_prefix_len[(batch,)](
            page_table, cache_seqlens, cu_seqlens_q, lcp, page_table.stride(0), BLK=1024,
        )
        _select_shared_prefix[(1,)](lcp, nlo, batch, BP2=triton.next_power_of_2(batch))
        _paged_gqa_prefix[(row_tiles, num_kv_heads, prefix_splits)](
            q, k, v, page_table, nlo, k_scale, v_scale, partial, lse,
            softmax_scale * _LOG2E, num_tokens * group, batch,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            partial.stride(0), partial.stride(1), partial.stride(2), lse.stride(0), lse.stride(1),
            G=group, D=head_dim, BM=_PREFILL_BM, BN=_PREFILL_BN,
            PSPLITS=prefix_splits, SLOT0=num_splits, num_warps=8, num_stages=1,
        )
    _paged_gqa_fwd[(batch, num_kv_heads, num_splits)](
        q, k, v, page_table, cache_seqlens, cu_seqlens_q, k_scale, v_scale, out, partial, lse,
        nlo, softmax_scale * _LOG2E,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        page_table.stride(0), out.stride(0), out.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2), lse.stride(0), lse.stride(1),
        G=group, D=head_dim, BM=packed_rows, BN=32, SPLITS=num_splits, WINDOW=window_left,
        CASCADE=cascade, num_warps=4, num_stages=2 if packed_rows >= 32 else 3,
    )
    if total_splits > 1:
        _paged_gqa_combine[(batch * max_seqlen_q, num_q_heads)](
            partial, lse, cu_seqlens_q, nlo, out,
            partial.stride(0), partial.stride(1), partial.stride(2),
            lse.stride(0), lse.stride(1), out.stride(0), out.stride(1),
            MAX_T=max_seqlen_q, SPLITS=total_splits, SP2=triton.next_power_of_2(total_splits),
            D=head_dim, SUFFIX_SPLITS=num_splits, num_warps=4,
        )
    return out
