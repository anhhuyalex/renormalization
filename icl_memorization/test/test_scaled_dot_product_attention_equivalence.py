import os
import sys
import unittest

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore


if torch is not None:
    # Allow importing icl_memorization/gpt.py as a plain module.
    _THIS_DIR = os.path.dirname(__file__)
    _ICL_MEM_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
    if _ICL_MEM_DIR not in sys.path:
        sys.path.insert(0, _ICL_MEM_DIR)

    import gpt  # noqa: E402


class TestScaledDotProductAttentionEquivalence(unittest.TestCase):
    def test_custom_matches_torch_builtin(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):  # pragma: no cover
            self.skipTest("torch.nn.functional.scaled_dot_product_attention not available")

        # Deterministic CPU setup.
        torch.manual_seed(0)
        device = torch.device("cpu")

        B, T = 2, 8
        num_heads = 3
        head_dim = 4
        vocab_size = 50
        len_context = 16
        num_mlp_layers = 1

        model = gpt.OneLayerAttention(
            len_context=len_context,
            num_heads=num_heads,
            num_hidden_features=head_dim,
            vocab_size=vocab_size,
            num_mlp_layers=num_mlp_layers,
        ).to(device)

        idx = torch.randint(0, vocab_size, (B, T), device=device)

        q, k, v, _C = model.get_qkv(idx, targets=None)

        # Built-in PyTorch implementation (used in forward()).
        out_builtin = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )

        # Custom implementation (used by get_attention_weights()).
        out_custom, _attn_weight = model.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        torch.testing.assert_close(out_custom, out_builtin, rtol=1e-5, atol=1e-6)

        # Also cover the exact wrapper method you call elsewhere.
        out_wrapper, _w = model.get_attention_weights(idx, targets=None)
        torch.testing.assert_close(out_wrapper, out_builtin, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

