"""Checks the SLURM task-id -> setting mapping used by main_grow.py / main_grow.sh.

- Tasks 0-71 must map to exactly what they did before the no-switch experiments were added.
- Tasks 63-71 must be the large model sizes with transfer_mode == "none".
- Tasks 72-107 must be the small-model no-switch experiments (no growth / width / head growth x 4 transfer modes).

Run from icl_memorization/:  python -m pytest tests/test_grow_settings.py
"""
import ast
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from grow_settings import select_by_task_id  # noqa: E402

# (task, grow_condition, num_heads, num_hidden_features, transfer_mode, seed_offset)
# Frozen from the original main_grow.py (git HEAD before the `none` mode was added).
ORIGINAL_0_TO_62 = [
    (0, 'base', 8, 32, 'attn_only', 0), (1, 'base', 8, 32, 'mlp_only', 0), (2, 'base', 8, 32, 'both', 0),
    (3, 'width_growth', 8, 64, 'attn_only', 0), (4, 'width_growth', 8, 64, 'mlp_only', 0), (5, 'width_growth', 8, 64, 'both', 0),
    (6, 'head_growth', 16, 32, 'attn_only', 0), (7, 'head_growth', 16, 32, 'mlp_only', 0), (8, 'head_growth', 16, 32, 'both', 0),
    (9, 'both_growth', 16, 64, 'attn_only', 0), (10, 'both_growth', 16, 64, 'mlp_only', 0), (11, 'both_growth', 16, 64, 'both', 0),
    (12, 'base', 8, 32, 'attn_only', 1), (13, 'base', 8, 32, 'mlp_only', 1), (14, 'base', 8, 32, 'both', 1),
    (15, 'width_growth', 8, 64, 'attn_only', 1), (16, 'width_growth', 8, 64, 'mlp_only', 1), (17, 'width_growth', 8, 64, 'both', 1),
    (18, 'head_growth', 16, 32, 'attn_only', 1), (19, 'head_growth', 16, 32, 'mlp_only', 1), (20, 'head_growth', 16, 32, 'both', 1),
    (21, 'both_growth', 16, 64, 'attn_only', 1), (22, 'both_growth', 16, 64, 'mlp_only', 1), (23, 'both_growth', 16, 64, 'both', 1),
    (24, 'base', 8, 32, 'attn_only', 2), (25, 'base', 8, 32, 'mlp_only', 2), (26, 'base', 8, 32, 'both', 2),
    (27, 'width_growth', 8, 64, 'attn_only', 2), (28, 'width_growth', 8, 64, 'mlp_only', 2), (29, 'width_growth', 8, 64, 'both', 2),
    (30, 'head_growth', 16, 32, 'attn_only', 2), (31, 'head_growth', 16, 32, 'mlp_only', 2), (32, 'head_growth', 16, 32, 'both', 2),
    (33, 'both_growth', 16, 64, 'attn_only', 2), (34, 'both_growth', 16, 64, 'mlp_only', 2), (35, 'both_growth', 16, 64, 'both', 2),
    (36, 'base_width_growth', 8, 64, 'attn_only', 0), (37, 'base_width_growth', 8, 64, 'mlp_only', 0), (38, 'base_width_growth', 8, 64, 'both', 0),
    (39, 'base_head_growth', 16, 32, 'attn_only', 0), (40, 'base_head_growth', 16, 32, 'mlp_only', 0), (41, 'base_head_growth', 16, 32, 'both', 0),
    (42, 'base_both_growth', 16, 64, 'attn_only', 0), (43, 'base_both_growth', 16, 64, 'mlp_only', 0), (44, 'base_both_growth', 16, 64, 'both', 0),
    (45, 'base_width_growth', 8, 64, 'attn_only', 1), (46, 'base_width_growth', 8, 64, 'mlp_only', 1), (47, 'base_width_growth', 8, 64, 'both', 1),
    (48, 'base_head_growth', 16, 32, 'attn_only', 1), (49, 'base_head_growth', 16, 32, 'mlp_only', 1), (50, 'base_head_growth', 16, 32, 'both', 1),
    (51, 'base_both_growth', 16, 64, 'attn_only', 1), (52, 'base_both_growth', 16, 64, 'mlp_only', 1), (53, 'base_both_growth', 16, 64, 'both', 1),
    (54, 'base_width_growth', 8, 64, 'attn_only', 2), (55, 'base_width_growth', 8, 64, 'mlp_only', 2), (56, 'base_width_growth', 8, 64, 'both', 2),
    (57, 'base_head_growth', 16, 32, 'attn_only', 2), (58, 'base_head_growth', 16, 32, 'mlp_only', 2), (59, 'base_head_growth', 16, 32, 'both', 2),
    (60, 'base_both_growth', 16, 64, 'attn_only', 2), (61, 'base_both_growth', 16, 64, 'mlp_only', 2), (62, 'base_both_growth', 16, 64, 'both', 2),
]

# Tasks 63-71 as launched earlier (frozen).
ORIGINAL_63_TO_71 = [
    (63, 'base_width_growth', 8, 64, 'none', 0), (64, 'base_head_growth', 16, 32, 'none', 0), (65, 'base_both_growth', 16, 64, 'none', 0),
    (66, 'base_width_growth', 8, 64, 'none', 1), (67, 'base_head_growth', 16, 32, 'none', 1), (68, 'base_both_growth', 16, 64, 'none', 1),
    (69, 'base_width_growth', 8, 64, 'none', 2), (70, 'base_head_growth', 16, 32, 'none', 2), (71, 'base_both_growth', 16, 64, 'none', 2),
]
ORIGINAL_0_TO_71 = ORIGINAL_0_TO_62 + ORIGINAL_63_TO_71

LARGE_SIZES = {(8, 64), (16, 32), (16, 64)}  # anything bigger than the 8h/32d base model


@pytest.mark.parametrize("task,cond,heads,hidden,mode,seed_offset", ORIGINAL_0_TO_71)
def test_tasks_0_to_71_unchanged(task, cond, heads, hidden, mode, seed_offset):
    s, so = select_by_task_id(task)
    assert (s["grow_condition"], s["num_heads"], s["num_hidden_features"], s["transfer_mode"], so) == \
        (cond, heads, hidden, mode, seed_offset)


def test_original_table_covers_0_to_71():
    assert [row[0] for row in ORIGINAL_0_TO_71] == list(range(72))


@pytest.mark.parametrize("task", range(63, 72))
def test_tasks_63_to_71_are_large_and_reinitialized(task):
    s, seed_offset = select_by_task_id(task)
    assert s["transfer_mode"] == "none"
    assert (s["num_heads"], s["num_hidden_features"]) in LARGE_SIZES
    assert s["grow_condition"].startswith("base_")  # base experiment: no growth
    assert s["is_base_experiment"] is True
    assert seed_offset in (0, 1, 2)


def test_tasks_63_to_71_cover_each_size_and_seed_once():
    seen = {(select_by_task_id(t)[0]["grow_condition"], select_by_task_id(t)[1]) for t in range(63, 72)}
    assert seen == {(c, so) for c in ("base_width_growth", "base_head_growth", "base_both_growth") for so in range(3)}


def test_reinit_runs_share_seeds_with_matching_base_runs():
    # Same size + same seed_offset as an existing run, so the first 50 epochs are identical (paired comparison).
    for t in range(63, 72):
        s, so = select_by_task_id(t)
        twins = [r for r in ORIGINAL_0_TO_62[36:] if r[1] == s["grow_condition"] and r[5] == so]
        assert len(twins) == 3


def test_nothing_else_uses_none():
    assert all(select_by_task_id(t)[0]["transfer_mode"] != "none" for t in range(63))


@pytest.mark.parametrize("task", range(72, 108))
def test_tasks_72_to_107_are_small_model_no_switch(task):
    s, seed_offset = select_by_task_id(task)
    assert s["grow_condition"] in ("noswitch_base", "noswitch_width_growth", "noswitch_head_growth")
    assert (s["num_heads"], s["num_hidden_features"]) in {(8, 32), (8, 64), (16, 32)}  # no both_growth
    assert s["transfer_mode"] in ("attn_only", "mlp_only", "both", "none")
    assert s["resample_at_switch"] is False
    assert s["is_base_experiment"] is False  # starts from the small 8h/32d model
    assert seed_offset in (0, 1, 2)


def test_tasks_72_to_107_cover_every_condition_mode_seed_once():
    seen = [(select_by_task_id(t)[0]["grow_condition"], select_by_task_id(t)[0]["transfer_mode"], select_by_task_id(t)[1])
            for t in range(72, 108)]
    expected = {(c, m, so) for c in ("noswitch_base", "noswitch_width_growth", "noswitch_head_growth")
                for m in ("attn_only", "mlp_only", "both", "none") for so in range(3)}
    assert len(seen) == len(set(seen)) == 36 and set(seen) == expected


def test_only_noswitch_tasks_skip_resampling():
    for t in range(72):
        assert select_by_task_id(t)[0].get("resample_at_switch", True) is True


def test_noswitch_size_matches_growth_experiment_sizes():
    # Same target sizes as the matching growth experiments (0-35), so results line up.
    sizes = {r[1]: (r[2], r[3]) for r in ORIGINAL_0_TO_62[:36]}
    for t in range(72, 108):
        s, _ = select_by_task_id(t)
        assert (s["num_heads"], s["num_hidden_features"]) == sizes[s["grow_condition"].replace("noswitch_", "")] \
            or s["grow_condition"] == "noswitch_base"


def test_main_grow_skips_resample_when_flag_false():
    with open(os.path.join(ROOT, "main_grow.py")) as f:
        src = f.read()
    assert 'args.resample_at_switch = selected_setting.get("resample_at_switch", True)' in src
    assert 'if getattr(args, "resample_at_switch", True):' in src


def test_sbatch_array_is_72_to_107():
    with open(os.path.join(ROOT, "main_grow.sh")) as f:
        m = re.search(r"^#SBATCH --array=(\d+)-(\d+)", f.read(), re.M)
    assert m is not None
    assert (int(m.group(1)), int(m.group(2))) == (72, 107)


def test_transfer_mode_none_is_accepted_by_cli():
    with open(os.path.join(ROOT, "main_grow.py")) as f:
        src = f.read()
    m = re.search(r"'--transfer_mode'.*?choices=\[(.*?)\]", src, re.S)
    assert m is not None and "'none'" in m.group(1)
    m = re.search(r"'--grow_condition'.*?choices=\[(.*?)\]", src, re.S)
    assert m is not None and all(f"'{c}'" in m.group(1) for c in ("noswitch_base", "noswitch_width_growth", "noswitch_head_growth"))


def test_none_mode_copies_nothing():
    """grow_onelayer_attention_weights in main_grow.py must leave a fresh target untouched for mode 'none'."""
    torch = pytest.importorskip("torch")
    import gpt

    with open(os.path.join(ROOT, "main_grow.py")) as f:
        tree = ast.parse(f.read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "grow_onelayer_attention_weights")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), "main_grow.py", "exec"), ns)
    grow = ns["grow_onelayer_attention_weights"]

    def make(h, d):
        return gpt.OneLayerAttention(len_context=8, num_heads=h, num_hidden_features=d, vocab_size=2,
                                     num_mlp_layers=3, num_hidden_features_mlp=h * d)

    torch.manual_seed(0)
    old = make(2, 4)
    new = make(2, 4)
    fresh = {k: v.clone() for k, v in new.state_dict().items()}
    out = grow(old, new, 2, 4, 2, 4, transfer_mode="none")
    for k, v in out.state_dict().items():
        assert torch.equal(v, fresh[k]), k
    # sanity: "both" at the same size does copy the old weights
    new2 = make(2, 4)
    out2 = grow(old, new2, 2, 4, 2, 4, transfer_mode="both")
    assert all(torch.equal(out2.state_dict()[k], v) for k, v in old.state_dict().items())
