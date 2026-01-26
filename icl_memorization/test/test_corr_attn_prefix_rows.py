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


def _make_strictly_causal_attn(B: int, H: int, N: int, *, seed: int = 0) -> torch.Tensor:
    """
    Create attention probabilities with shape (B,H,N,N) where row t has support only on :t
    (strictly lower-triangular), with a special-case for t=0 so the row is well-defined.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    logits = torch.randn(B, H, N, N, generator=g)

    # Strictly causal: allow j < i. For i=0, allow j=0 to avoid all -inf.
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool), diagonal=-1)
    mask[0, 0] = True

    logits = logits.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(logits, dim=-1)
    return attn


def _pearson_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation for 1D tensors of the same length, matching utils.pearson_corr_lastdim.
    """
    x0 = x - x.mean()
    y0 = y - y.mean()
    x0 = x0 / (x0.std(unbiased=False) + eps)
    y0 = y0 / (y0.std(unbiased=False) + eps)
    return (x0 * y0).mean()


class TestCorrAttnPrefixRows(unittest.TestCase):
    def test_matches_naive_loop(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")
        B, H, N = 2, 3, 12
        attn = _make_strictly_causal_attn(B, H, N, seed=123)
        step = 5
        eps = 1e-8

        corr = utils.corr_attn_prefix_rows(attn, step=step, eps=eps, chunk_t=None)
        positions = torch.arange(0, N, step)  # s_positions
        P_s = positions.numel()
        P_t = int(((positions + 1) < N).sum().item())

        self.assertEqual(tuple(corr.shape), (B, H, P_s, P_t))

        # Entries with j < i should be NaN by default.
        ii = torch.arange(P_s).view(P_s, 1)
        jj = torch.arange(P_t).view(1, P_t)
        below_diag = jj < ii
        self.assertTrue(torch.isnan(corr[..., below_diag]).all().item())

        # Compare each valid (s,t) entry to a naive loop.
        # Entry (i,j) corresponds to s = positions[i], t = positions[j] + 1, only for j >= i.
        for i in range(P_s):
            s = int(positions[i].item())
            if s <= 0:
                continue
            for j in range(i + 1, P_t):
                t = int(positions[j].item()) + 1
                if t >= N:
                    continue
                for b in range(B):
                    for h in range(H):
                        x = attn[b, h, s, :s]
                        y = attn[b, h, t, :s]
                        x = x / (x.sum() + eps)
                        y = y / (y.sum() + eps)
                        expected = _pearson_1d(x, y, eps=eps)
                        got = corr[b, h, i, j]
                        torch.testing.assert_close(
                            got, expected, rtol=1e-5, atol=1e-6, msg=f"b={b}, h={h}, s={s}, t={t}"
                        )

    def test_chunking_matches_unchunked(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")
        B, H, N = 4, 2, 25
        attn = _make_strictly_causal_attn(B, H, N, seed=7)

        corr0 = utils.corr_attn_prefix_rows(attn, step=5, chunk_t=None)
        corr1 = utils.corr_attn_prefix_rows(attn, step=5, chunk_t=1)
        corr2 = utils.corr_attn_prefix_rows(attn, step=5, chunk_t=2)

        # NaNs should be in the same places; compare only the valid upper-triangular entries.
        valid = ~torch.isnan(corr0)
        torch.testing.assert_close(corr1[valid], corr0[valid], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(corr2[valid], corr0[valid], rtol=1e-6, atol=1e-7)

    def test_return_nan_matrix_false_fills_zero(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")
        B, H, N = 2, 1, 12
        attn = _make_strictly_causal_attn(B, H, N, seed=99)

        corr = utils.corr_attn_prefix_rows(attn, step=5, return_nan_matrix=False)
        positions = torch.arange(0, N, 5)
        P_s = positions.numel()
        P_t = int(((positions + 1) < N).sum().item())

        ii = torch.arange(P_s).view(P_s, 1)
        jj = torch.arange(P_t).view(1, P_t)
        tri = jj >= ii

        # Invalid entries should be exactly 0.0 when return_nan_matrix=False.
        self.assertTrue((corr[..., ~tri] == 0).all().item())

    def test_positions_override(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch not installed in this environment")
        B, H, N = 2, 2, 10
        attn = _make_strictly_causal_attn(B, H, N, seed=5)
        positions = torch.tensor([0, 3, 7], dtype=torch.long)

        corr = utils.corr_attn_prefix_rows(attn, positions=positions)
        self.assertEqual(tuple(corr.shape), (B, H, positions.numel(), positions.numel()))


if __name__ == "__main__":
    unittest.main()

