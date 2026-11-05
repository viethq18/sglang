"""Full-chain MTP preserves generic tree topology without probability ranking."""

import json
import tempfile
import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import frozen_kv_mtp_worker_v2 as module
from sglang.srt.speculative.adaptive_goodput_policy import GoodputAdaptivePolicy
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput
from sglang.srt.speculative.spec_utils import select_top_k_tokens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFrozenKVMTPChain(CustomTestCase):
    @torch.compiler.set_stance("force_eager")
    def check_tree(self, steps, topk, width):
        bs = 3
        generator = torch.Generator().manual_seed(17)
        logits = [
            torch.randn((bs * topk, 37), generator=generator) * 3 for _ in range(steps)
        ]
        hidden = [
            torch.randn((bs * topk, 8), generator=generator) for _ in range(steps)
        ]
        calls = []

        def forward(batch):
            i = len(calls)
            calls.append(
                (batch.input_ids.clone(), batch.spec_info.hidden_states.clone())
            )
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=logits[i], hidden_states=hidden[i]
                )
            )

        seed = FrozenKVMTPDraftInput(
            bonus_tokens=torch.arange(bs), hidden_states=torch.ones(bs, 8)
        )
        batch = SimpleNamespace(
            spec_info=seed, needs_forward_metadata_init=lambda: False
        )
        worker = SimpleNamespace(
            topk=topk,
            speculative_num_steps=steps,
            speculative_num_draft_tokens=width,
            draft_model_runner=SimpleNamespace(forward=forward),
            draft_attn_backend=None,
            _set_positions=lambda b: None,
            _target_kv_pool_view=lambda b: nullcontext(),
        )
        with patch.object(
            module, "forward_context", lambda context, **kwargs: nullcontext()
        ):
            actual = module.FrozenKVMTPDraftWorker.draft_forward(worker, batch)
        scores = None
        score_list = []
        token_list = []
        parent_list = []
        expected_calls = []
        for i in range(steps):
            current_logits = logits[i][::topk] if i == 0 else logits[i]
            current_hidden = hidden[i][::topk] if i == 0 else hidden[i]
            probabilities = torch.softmax(current_logits, dim=-1)
            topk_probs, indices = torch.topk(probabilities, topk, dim=-1)
            inputs, selected_hidden, scores, tree = select_top_k_tokens(
                i, topk_probs, indices, current_hidden, scores, topk
            )
            score_list.append(tree[0])
            token_list.append(tree[1])
            parent_list.append(tree[2])
            if i < steps - 1:
                expected_calls.append((inputs, selected_hidden))
        expected = organize_draft_results(score_list, token_list, parent_list, width)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            # The CUDA tree builder checks the packed-row layout, whereas
            # the XPU implementation accepts expanded views as well.
            self.assertTrue(a.is_contiguous())
        for a, b in zip(calls[1:], expected_calls):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_full_chain_preserves_tree_and_recurrent_inputs(self):
        # Parent layout has a special one-step boundary; later steps must
        # preserve each request's hidden state without a frontier gather.
        for steps in (1, 2, 3, 5):
            with self.subTest(steps=steps):
                self.check_tree(steps, 1, steps + 1)

    def test_pruned_and_branching_trees_keep_probability_ranking(self):
        for topk, width in ((1, 3), (2, 5)):
            with self.subTest(topk=topk, width=width):
                self.check_tree(3, topk, width)


class _Event:
    def __init__(self, ms=None):
        self.ms = ms

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        if end.ms is None:
            raise AssertionError("a non-leader rank read its own GPU timing")
        return end.ms


class _Rank0BroadcastGroup:
    """Collective stand-in: rank 0 publishes, the other rank receives it."""

    def __init__(self, rank, wire):
        self.rank_in_group = rank
        self.wire = wire

    def broadcast_object(self, obj, src=0):
        if self.rank_in_group == src:
            self.wire.append(obj)
            return obj
        return self.wire.pop(0)


class TestTensorParallelStepTiming(CustomTestCase):
    def _policy(self):
        cfg = {
            "policy": "goodput",
            "1": {"candidate_steps": [1, 3]},
            "min_samples": 1,
            "warmup_samples": 0,
            "alpha_halflife": 1e12,
            "explore_interval": 10**9,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(cfg, f)
            f.flush()
            return GoodputAdaptivePolicy(initial_steps=3, cfg_path=f.name)

    def _spans(self, n, cost_ms):
        # Alternating k=1 / k=3 decode steps at batch size 1.
        spans = deque()
        for i in range(n):
            k = 1 if i % 2 else 3
            ms = None if cost_ms is None else cost_ms[k]
            spans.append([k, 1, _Event(), _Event(ms)])
        return spans

    def test_ranks_with_diverging_local_timings_choose_rank0_steps(self):
        # alpha prior 0.5: E[1] = 1.5, E[3] = 1.875. Rank 0 times favour k=3;
        # rank 1 would favour k=1 (25 ms for k=3) if it trusted its own clock.
        wire = []
        picks = []
        for rank, cost_ms in ((0, {1: 10.0, 3: 10.0}), (1, None)):
            policy = self._policy()
            pending, free = self._spans(20, cost_ms), []
            module._observe_synced_step_times(
                pending, free, policy, _Rank0BroadcastGroup(rank, wire)
            )
            self.assertEqual(len(pending), 4)
            self.assertEqual(len(free), 32)
            picks.append(policy.get_steps_for_batch(1))
        self.assertEqual(picks, [3, 3])
        self.assertEqual(wire, [])

    def test_no_collective_until_a_full_batch_plus_lag_is_pending(self):
        wire = []
        policy = self._policy()
        pending, free = self._spans(19, {1: 10.0, 3: 10.0}), []
        module._observe_synced_step_times(
            pending, free, policy, _Rank0BroadcastGroup(0, wire)
        )
        self.assertEqual((len(pending), free, wire), (19, [], []))
        self.assertIsNone(policy.goodput(3, 1))


if __name__ == "__main__":
    unittest.main()
