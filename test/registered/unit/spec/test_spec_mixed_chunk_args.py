import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_spec_args(algorithm: str) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # speculative hook directly (same pattern as the unit/server_args tests).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.device = "cuda"
    args.enable_mixed_chunk = True
    args.max_running_requests = 64
    args.page_size = 1
    args.speculative_num_steps = 3
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 4
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Gemma4ForConditionalGeneration"],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    return args


class TestSpecMixedChunkArgs(CustomTestCase):
    def test_frozen_kv_mtp_keeps_mixed_chunk(self):
        args = _make_spec_args("FROZEN_KV_MTP")

        handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "enable_mixed_chunk"))

    def test_ngram_still_disables_mixed_chunk(self):
        args = _make_spec_args("NGRAM")

        handle_speculative_decoding(args)

        self.assertFalse(resolution_result(args, "enable_mixed_chunk"))


if __name__ == "__main__":
    unittest.main()
