import json
import tempfile
import unittest

from sglang.srt.speculative.adaptive_goodput_policy import GoodputAdaptivePolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _policy(cfg: dict, initial_steps: int = 3) -> GoodputAdaptivePolicy:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump({"policy": "goodput", **cfg}, f)
        f.flush()
        return GoodputAdaptivePolicy(initial_steps=initial_steps, cfg_path=f.name)


# No warmup, one sample makes a cost usable, no forgetting, no exploration.
_FAST = {
    "min_samples": 1,
    "warmup_samples": 0,
    "alpha_halflife": 1e12,
    "explore_interval": 10**9,
}


class TestAcceptanceModel(CustomTestCase):
    def test_censored_per_position_estimate_across_mixed_draft_lengths(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "alpha_prior_weight": 1e-9})
        # k=3: position i is tried only by requests that accepted i-1 drafts.
        p.observe_accepts([3, 1, 0, 2], num_steps=3)
        alpha = p.alpha()
        self.assertAlmostEqual(alpha[1], 3 / 4)
        self.assertAlmostEqual(alpha[2], 2 / 3)
        self.assertAlmostEqual(alpha[3], 1 / 2)

        # k=1 batches inform position 1 only.
        p.observe_accepts([1, 0], num_steps=1)
        alpha = p.alpha()
        self.assertAlmostEqual(alpha[1], 4 / 6)
        self.assertAlmostEqual(alpha[2], 2 / 3)
        self.assertAlmostEqual(alpha[3], 1 / 2)

    def test_untried_positions_inherit_previous_position(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "alpha_prior": 0.5})
        self.assertEqual(p.alpha()[1:], [0.5, 0.5, 0.5])

        p.observe_accepts([1] * 8, num_steps=1)
        alpha = p.alpha()
        # (8 successes + 4 * 0.5 prior) / (8 trials + 4).
        self.assertAlmostEqual(alpha[1], 10 / 12)
        self.assertAlmostEqual(alpha[2], alpha[1])
        self.assertAlmostEqual(alpha[3], alpha[1])

    def test_accept_counts_above_draft_length_are_clamped(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "alpha_prior_weight": 1e-9})
        p.observe_accepts([5, 0], num_steps=1)
        self.assertAlmostEqual(p.alpha()[1], 1 / 2)

    def test_zero_step_batches_do_not_move_alpha(self):
        p = _policy({"1": {"candidate_steps": [0, 3]}, **_FAST})
        before = list(p.alpha())
        p.observe_accepts([0, 0, 0], num_steps=0)
        self.assertEqual(p.alpha(), before)

    def test_expected_tokens(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "alpha_prior": 0.5})
        self.assertEqual(p.expected_tokens(0), 1.0)
        self.assertAlmostEqual(p.expected_tokens(3), 1 + 0.5 + 0.25 + 0.125)

    def test_old_samples_decay(self):
        p = _policy(
            {"1": {"candidate_steps": [1, 3]}, **_FAST, "alpha_halflife": 4,
             "alpha_prior_weight": 1e-9}
        )
        p.observe_accepts([0] * 4, num_steps=1)
        p.observe_accepts([1] * 4, num_steps=1)
        # The older batch weighs half: 4 / (2 + 4).
        self.assertAlmostEqual(p.alpha()[1], 4 / 6)


class TestCostModel(CustomTestCase):
    def test_warmup_samples_are_dropped(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "warmup_samples": 2})
        p.observe_step_time(3, 1, 100.0)
        p.observe_step_time(3, 1, 100.0)
        self.assertIsNone(p.goodput(3, 1))
        p.observe_step_time(3, 1, 10.0)
        self.assertAlmostEqual(p.goodput(3, 1), p.expected_tokens(3) / 10.0)

    def test_min_samples_gate(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "min_samples": 2})
        p.observe_step_time(3, 1, 10.0)
        self.assertIsNone(p.goodput(3, 1))
        p.observe_step_time(3, 1, 10.0)
        self.assertIsNotNone(p.goodput(3, 1))

    def test_isolated_outlier_dropped_persistent_shift_adopted(self):
        p = _policy(
            {"1": {"candidate_steps": [1, 3]}, **_FAST, "cost_ema": 0.5,
             "outlier_ratio": 4.0, "outlier_streak": 3}
        )
        p.observe_step_time(3, 1, 10.0)
        p.observe_step_time(3, 1, 50.0)
        p.observe_step_time(3, 1, 50.0)
        # A normal sample breaks the outlier run.
        p.observe_step_time(3, 1, 10.0)
        self.assertAlmostEqual(p._costs[(3, 1)].ema_ms, 10.0)

        for _ in range(3):
            p.observe_step_time(3, 1, 50.0)
        self.assertAlmostEqual(p._costs[(3, 1)].ema_ms, 50.0)

    def test_step_time_keyed_by_padded_cuda_graph_bs(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST})
        p.set_cuda_graph_bs([1, 2, 4, 8])
        p.observe_step_time(3, 3, 10.0)
        self.assertIsNotNone(p.goodput(3, 4))
        self.assertIsNone(p.goodput(3, 3))
        self.assertIsNone(p.goodput(3, 8))


class TestSelection(CustomTestCase):
    def _warm(self, p, bs, costs):
        """Drive decisions until every candidate has a cost; return the picks."""
        picks = []
        for _ in range(len(costs)):
            k = p.get_steps_for_batch(bs)
            picks.append(k)
            p.observe_step_time(k, bs, costs[k])
        return picks

    def test_cold_start_measures_current_then_each_candidate(self):
        p = _policy({"1": {"candidate_steps": [1, 3, 5]}, **_FAST})
        picks = self._warm(p, 1, {1: 10.0, 3: 10.0, 5: 20.0})
        self.assertEqual(picks, [3, 1, 5])

    def test_cold_start_repeats_until_estimate_is_usable(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "warmup_samples": 1})
        picks = []
        for _ in range(4):
            k = p.get_steps_for_batch(1)
            picks.append(k)
            p.observe_step_time(k, 1, 10.0)
        self.assertEqual(picks, [3, 3, 1, 1])

    def test_argmax_of_expected_tokens_over_step_time(self):
        # alpha = 0.5 everywhere: E[1] = 1.5, E[3] = 1.875.
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST})
        self._warm(p, 1, {1: 10.0, 3: 10.0})
        self.assertEqual(p.get_steps_for_batch(1), 3)

        p2 = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST})
        self._warm(p2, 1, {1: 10.0, 3: 25.0})
        self.assertEqual(p2.get_steps_for_batch(1), 1)

    def test_switch_needs_margin(self):
        e1, e3 = 1.5, 1.875
        gp3 = e3 / 10.0
        for gain, expected in ((1.01, 3), (1.05, 1)):
            p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "switch_margin": 0.02})
            self._warm(p, 1, {1: e1 / (gp3 * gain), 3: 10.0})
            self.assertEqual(p.get_steps_for_batch(1), expected, gain)

    def test_acceptance_drives_the_choice(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST, "alpha_prior_weight": 1e-9})
        self._warm(p, 1, {1: 10.0, 3: 14.0})
        p.observe_accepts([3] * 50, num_steps=3)
        self.assertEqual(p.get_steps_for_batch(1), 3)
        p.observe_accepts([0] * 1000, num_steps=3)
        self.assertEqual(p.get_steps_for_batch(1), 1)

    def test_periodic_exploration_alternates_least_recent_neighbour(self):
        p = _policy(
            {"1": {"candidate_steps": [1, 3, 5]}, **_FAST, "explore_interval": 4}
        )
        costs = {1: 10.0, 3: 10.0, 5: 20.0}
        picks = []
        for _ in range(12):
            k = p.get_steps_for_batch(1)
            picks.append(k)
            p.observe_step_time(k, 1, costs[k])
        self.assertEqual(picks, [3, 1, 5, 1, 3, 3, 3, 5, 3, 3, 3, 1])

    def test_buckets_and_slots_are_independent(self):
        p = _policy(
            {
                "1": {"candidate_steps": [1, 3]},
                "16": {"candidate_steps": [0, 1]},
                "64": {"candidate_steps": [0]},
                **_FAST,
            }
        )
        p.set_cuda_graph_bs([1, 2, 4, 8, 16, 32, 64])
        self._warm(p, 1, {1: 10.0, 3: 25.0})
        self.assertEqual(p.get_steps_for_batch(1), 1)
        # bs=2 is a fresh bucket of slot "1": cold start from initial steps.
        self.assertEqual(p.get_steps_for_batch(2), 3)
        # Slot "16" does not offer the initial steps; cold start from its first.
        self.assertEqual(p.get_steps_for_batch(16), 0)
        self.assertEqual(p.get_steps_for_batch(64), 0)
        self.assertEqual(p.get_steps_for_batch(100), 0)

    def test_on_verify_complete_never_switches(self):
        p = _policy({"1": {"candidate_steps": [1, 3]}, **_FAST})
        self.assertIsNone(p.on_verify_complete([0, 0], batch_size=2))


if __name__ == "__main__":
    unittest.main()
