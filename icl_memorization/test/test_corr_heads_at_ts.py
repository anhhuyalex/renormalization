import os
import sys
import unittest

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore


if torch is not None:
    # Allow importing icl_memorization/utils.py as a plain module.
    _THIS_DIR = os.path.dirname(__file__)
    _ICL_MEM_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
    if _ICL_MEM_DIR not in sys.path:
        sys.path.insert(0, _ICL_MEM_DIR)

    import utils  # noqa: E402


def _make_random_attn(B: int, H: int, N: int, *, seed: int = 0) -> torch.Tensor:
    """
    Create attention probabilities with shape (B,H,N,N) without assuming causality.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    logits = torch.randn(B, H, N, N, generator=g)
    return torch.softmax(logits, dim=-1)


class TestCorrHeadsAtTs(unittest.TestCase):
    def test_ts_clipping_and_shape(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")

        # Here H > N, and default ts is arange(0, H, 5) -> [0,5,...]
        # but we must clip to < N to avoid out-of-bounds on the sequence axis.
        B, H, N = 2, 11, 4
        attn = _make_random_attn(B, H, N, seed=1)

        corr, ts_valid = utils.corr_heads_at_ts(attn, step=5, return_ts=True)
        self.assertTrue((ts_valid < N).all().item())
        self.assertEqual(tuple(corr.shape), (B, int(ts_valid.numel()), H, H))

        # For t<=1, function leaves fill value (NaN by default).
        if (ts_valid <= 1).any().item():
            ti = int((ts_valid <= 1).nonzero(as_tuple=False)[0].item())
            self.assertTrue(torch.isnan(corr[:, ti]).all().item())

    def test_matches_naive_pairwise_pearson(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")

        B, H, N = 2, 4, 10
        attn = _make_random_attn(B, H, N, seed=2)
        ts = torch.tensor([1, 5, 9], dtype=torch.long)  # includes t=1 edge

        corr, ts_valid = utils.corr_heads_at_ts(attn, ts=ts, return_ts=True, eps=1e-8)

        # Check t=1 remains NaN (undefined: length 1).
        if (ts_valid == 1).any().item():
            ti = int((ts_valid == 1).nonzero(as_tuple=False)[0].item())
            self.assertTrue(torch.isnan(corr[:, ti]).all().item())

        # Check a valid t against naive pearson for a few entries.
        for ti, t in enumerate(ts_valid.tolist()):
            if t <= 1:
                continue
            X = attn[:, :, t, :t]  # (B,H,t)
            for b in range(B):
                for i in range(H):
                    for j in range(H):
                        expected = utils.pearson_corr_lastdim(X[b, i], X[b, j])
                        got = corr[b, ti, i, j]
                        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
            break  # one valid t is enough

    def test_identical_heads_give_corr_one(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")

        B, H, N = 1, 6, 8
        attn = _make_random_attn(B, H, N, seed=3)

        # Force all heads to have the same prefix row at a chosen t.
        t = 6
        v = torch.linspace(0.0, 1.0, steps=t)  # non-constant to avoid zero std
        v = v / v.sum()
        attn[:, :, t, :t] = v  # broadcast to (B,H,t)

        corr, ts_valid = utils.corr_heads_at_ts(attn, ts=torch.tensor([t]), return_ts=True)
        self.assertEqual(ts_valid.tolist(), [t])

        # With identical vectors across heads, Pearson correlation should be ~1.
        C = corr[0, 0]  # (H,H)
        ones = torch.ones_like(C)
        torch.testing.assert_close(C, ones, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

