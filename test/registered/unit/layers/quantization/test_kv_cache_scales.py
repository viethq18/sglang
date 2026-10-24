"""Checkpoint KV scales vs. the KV cache dtype — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.kv_cache import (
    BaseKVCacheMethod,
    drop_kv_scales_for_unquantized_cache,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.test.test_utils import CustomTestCase


def _attn_with_loaded_scales(k_scale: float, v_scale: float) -> RadixAttention:
    attn = RadixAttention(
        num_heads=2, head_dim=8, scaling=1.0, num_kv_heads=1, layer_id=0
    )
    attn.quant_method = BaseKVCacheMethod(quant_config=None)
    attn.quant_method.create_weights(attn)
    attn.k_scale.data.fill_(k_scale)
    attn.v_scale.data.fill_(v_scale)
    attn.quant_method.process_weights_after_loading(attn)
    return attn


class TestDropKVScalesForUnquantizedCache(CustomTestCase):
    def test_bf16_cache_drops_scales(self):
        """A bf16 cache stores K/V unscaled; keeping k_scale would descale reads."""
        attn = _attn_with_loaded_scales(0.004, 0.04)
        drop_kv_scales_for_unquantized_cache(attn, torch.bfloat16)
        self.assertIsNone(attn.k_scale)
        self.assertIsNone(attn.v_scale)
        self.assertIsNone(attn.k_scale_float)
        self.assertIsNone(attn.v_scale_float)

    def test_fp8_cache_keeps_scales(self):
        attn = _attn_with_loaded_scales(0.004, 0.04)
        drop_kv_scales_for_unquantized_cache(attn, torch.float8_e4m3fn)
        self.assertAlmostEqual(attn.k_scale_float, 0.004, places=6)
        self.assertAlmostEqual(attn.v_scale_float, 0.04, places=6)
        self.assertAlmostEqual(attn.k_scale.item(), 0.004, places=6)


if __name__ == "__main__":
    unittest.main()
