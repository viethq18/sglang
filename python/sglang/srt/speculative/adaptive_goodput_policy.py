"""Goodput-driven adaptive speculative policy.

Picks the draft length ``k`` that maximises measured goodput for the current
batch size: ``E[tokens | k] / T(k, bs)``.

* ``E[tokens | k] = 1 + sum_{j<=k} prod_{i<=j} alpha_i``. The per-position
  acceptance ``alpha_i`` is estimated from verify results with censoring:
  position ``i`` was tried only when the first ``i-1`` drafts were accepted.
* ``T(k, bs)`` is an EMA of the measured GPU time of one decode step, keyed by
  the CUDA-graph batch size the step was padded to.

Selected with ``"policy": "goodput"`` in ``--speculative-adaptive-config``.
Candidate steps per batch size use the same ``"<bs>": {"candidate_steps"}``
slots as the default acceptance policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    _load_adaptive_config,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_GOODPUT_DEFAULTS = {
    # A non-current neighbour of the bucket's k is run once every this many
    # decisions, so stale cost estimates get refreshed.
    "explore_interval": 64,
    # Relative goodput gain needed to move a bucket away from its current k.
    "switch_margin": 0.02,
    # EMA weight of a new step-time sample.
    "cost_ema": 0.2,
    # Half-life, in request samples, of the acceptance counts.
    "alpha_halflife": 2000.0,
    # Pseudo-count weight pulling alpha_i towards alpha_{i-1}.
    "alpha_prior_weight": 4.0,
    # Prior for alpha_1 before any data arrives.
    "alpha_prior": 0.5,
    # Step-time samples needed per (k, bucket) before the estimate is used.
    "min_samples": 2,
    # Leading step-time samples dropped per (k, bucket): first graph replays
    # and lazily captured post-verify graphs.
    "warmup_samples": 2,
    # A sample above ``outlier_ratio * EMA`` is dropped unless it is the
    # ``outlier_streak``-th in a row (a real regime change).
    "outlier_ratio": 4.0,
    "outlier_streak": 3,
}


@dataclass
class _CostStat:
    ema_ms: float = 0.0
    num_samples: int = 0
    num_seen: int = 0
    outlier_run: int = 0
    # Bucket decision index at which this k was last chosen.
    last_chosen: int = -1


@dataclass
class _BucketState:
    current_steps: int
    num_decisions: int = 0


class GoodputAdaptivePolicy(AdaptiveSpeculativeParams):
    """Per-bucket argmax of expected tokens over measured step time."""

    needs_step_timing = True

    def __init__(self, initial_steps: int, cfg_path: str | None = None):
        super().__init__(initial_steps, cfg_path)
        cfg, _ = _load_adaptive_config(cfg_path)
        knobs = {k: cfg.get(k, v) for k, v in _GOODPUT_DEFAULTS.items()}
        self.explore_interval = int(knobs["explore_interval"])
        self.switch_margin = float(knobs["switch_margin"])
        self.cost_ema = float(knobs["cost_ema"])
        self.alpha_decay = 0.5 ** (1.0 / float(knobs["alpha_halflife"]))
        self.alpha_prior_weight = float(knobs["alpha_prior_weight"])
        self.alpha_prior = float(knobs["alpha_prior"])
        self.min_samples = int(knobs["min_samples"])
        self.warmup_samples = int(knobs["warmup_samples"])
        self.outlier_ratio = float(knobs["outlier_ratio"])
        self.outlier_streak = int(knobs["outlier_streak"])
        self.initial_steps = initial_steps

        max_k = max(self.candidate_steps)
        # Index 0 unused; positions are 1..max_k.
        self._trials = [0.0] * (max_k + 1)
        self._succ = [0.0] * (max_k + 1)
        self._alpha_cache: list[float] | None = None
        self._costs: dict[tuple[int, int], _CostStat] = {}
        self._buckets: dict[int, _BucketState] = {}

        log_info_on_rank0(
            logger,
            f"GoodputAdaptivePolicy initialized: candidate_steps="
            f"{self.candidate_steps}, knobs={knobs}",
        )

    # ------------------------------------------------------------------
    # Acceptance model
    # ------------------------------------------------------------------
    def observe_accepts(self, num_correct_drafts_per_req: list[int], num_steps: int):
        """Record per-request accepted draft counts of a batch drafted with
        ``num_steps`` tokens. ``num_steps == 0`` carries no information."""
        if num_steps <= 0 or not num_correct_drafts_per_req:
            return
        max_k = len(self._trials) - 1
        k = min(num_steps, max_k)
        hist = [0] * (k + 1)
        for c in num_correct_drafts_per_req:
            hist[min(max(int(c), 0), k)] += 1
        n = len(num_correct_drafts_per_req)
        decay = self.alpha_decay**n
        trials, succ = self._trials, self._succ
        for i in range(1, max_k + 1):
            trials[i] *= decay
            succ[i] *= decay
        # suffix[i] = #requests with c >= i.
        suffix = 0
        for i in range(k, 0, -1):
            suffix += hist[i]
            succ[i] += suffix
            # Tried iff c >= i - 1.
            trials[i] += suffix + hist[i - 1]
        self._alpha_cache = None

    def alpha(self) -> list[float]:
        """``alpha[i]`` for positions ``1..max_k`` (index 0 is unused)."""
        if self._alpha_cache is None:
            m = self.alpha_prior_weight
            out = [0.0] * len(self._trials)
            prev = self.alpha_prior
            for i in range(1, len(self._trials)):
                prev = (self._succ[i] + m * prev) / (self._trials[i] + m)
                out[i] = prev
            self._alpha_cache = out
        return self._alpha_cache

    def expected_tokens(self, num_steps: int) -> float:
        alpha = self.alpha()
        total, prod = 1.0, 1.0
        for i in range(1, num_steps + 1):
            prod *= alpha[i]
            total += prod
        return total

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------
    def observe_step_time(self, num_steps: int, batch_size: int, ms: float):
        bucket = self._pad_to_cuda_graph_bs(batch_size)
        stat = self._costs.setdefault((num_steps, bucket), _CostStat())
        stat.num_seen += 1
        if stat.num_seen <= self.warmup_samples:
            return
        if stat.num_samples == 0:
            stat.ema_ms = ms
            stat.num_samples = 1
            return
        if ms > self.outlier_ratio * stat.ema_ms:
            stat.outlier_run += 1
            if stat.outlier_run < self.outlier_streak:
                return
            # Persistent shift: restart the estimate from this sample.
            stat.ema_ms = ms
        else:
            stat.ema_ms += self.cost_ema * (ms - stat.ema_ms)
        stat.outlier_run = 0
        stat.num_samples += 1

    def goodput(self, num_steps: int, bucket: int) -> float | None:
        stat = self._costs.get((num_steps, bucket))
        if stat is None or stat.num_samples < self.min_samples:
            return None
        return self.expected_tokens(num_steps) / max(stat.ema_ms, 1e-6)

    # ------------------------------------------------------------------
    # AdaptiveSpecPolicy
    # ------------------------------------------------------------------
    def get_steps_for_batch(self, batch_size: int) -> int:
        bucket = self._pad_to_cuda_graph_bs(batch_size)
        cands = self._slots[self._find_closest_bs(bucket)].candidate_steps
        if len(cands) == 1:
            return cands[0]

        state = self._buckets.get(bucket)
        if state is None:
            start = self.initial_steps if self.initial_steps in cands else cands[0]
            state = self._buckets[bucket] = _BucketState(current_steps=start)
        if state.current_steps not in cands:
            state.current_steps = cands[0]
        state.num_decisions += 1
        chosen = self._choose(bucket, cands, state)
        self._costs.setdefault((chosen, bucket), _CostStat()).last_chosen = (
            state.num_decisions
        )
        return chosen

    def _choose(self, bucket: int, cands: list[int], state: _BucketState) -> int:
        def ready(k: int) -> bool:
            stat = self._costs.get((k, bucket))
            return stat is not None and stat.num_samples >= self.min_samples

        # Cold start: measure every candidate, current one first.
        if not ready(state.current_steps):
            return state.current_steps
        for k in cands:
            if not ready(k):
                return k

        cur = state.current_steps
        best = max(cands, key=lambda k: self.goodput(k, bucket))
        if best != cur and self.goodput(best, bucket) > self.goodput(
            cur, bucket
        ) * (1.0 + self.switch_margin):
            state.current_steps = cur = best

        if state.num_decisions % self.explore_interval == 0:
            idx = cands.index(cur)
            neighbours = [cands[j] for j in (idx - 1, idx + 1) if 0 <= j < len(cands)]
            return min(neighbours, key=lambda k: self._costs[(k, bucket)].last_chosen)
        return cur

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        # Switching happens per step in ``get_steps_for_batch``; acceptance is
        # fed through ``observe_accepts`` with the batch's own draft length.
        return None
