"""Experiment settings for main_grow.py, and the mapping from SLURM array task id to setting.

Task layout:
    0-35   growth experiments: start at 8h/32d, grow at the data switch (12 settings x 3 seeds)
    36-62  base experiments at larger sizes, partial transfer (9 settings x 3 seeds)
    63-71  base experiments at larger sizes, re-initialized with no transfer (3 settings x 3 seeds)
    72-107 no-switch experiments: start at 8h/32d, grow (or not) at epoch 50, but KEEP THE SAME SEQUENCES
           (3 sizes x 4 transfer modes = 12 settings x 3 seeds)
"""

NUM_SEEDS = 3
FIRST_BASE_TASK = 36
FIRST_NOSWITCH_TASK = 72
transfer_modes = ["attn_only", "mlp_only", "both"]

# 1. Growth settings (Tasks 0 to 35: start at 8h/32d, grow at epoch 50 to target size)
arch_conditions = [
    {"name": "base", "num_heads": 8, "num_hidden_features": 32, "desc": "base_8h_32d"},
    {"name": "width_growth", "num_heads": 8, "num_hidden_features": 64, "desc": "width_growth_8h_64d"},
    {"name": "head_growth", "num_heads": 16, "num_hidden_features": 32, "desc": "head_growth_16h_32d"},
    {"name": "both_growth", "num_heads": 16, "num_hidden_features": 64, "desc": "both_growth_16h_64d"},
]

grow_settings = []
for arch in arch_conditions:
    for t_mode in transfer_modes:
        grow_settings.append({
            "grow_condition": arch["name"],
            "num_heads": arch["num_heads"],
            "num_hidden_features": arch["num_hidden_features"],
            "transfer_mode": t_mode,
            "desc": f"{arch['desc']}_{t_mode}",
            "is_base_experiment": (arch["name"] == "base"),
        })

# 2. Base experiment settings for larger models (Tasks 36 to 62: start directly at target size from epoch 0)
base_arch_conditions = [
    {"name": "base_width_growth", "num_heads": 8, "num_hidden_features": 64, "desc": "base_width_8h_64d"},
    {"name": "base_head_growth", "num_heads": 16, "num_hidden_features": 32, "desc": "base_head_16h_32d"},
    {"name": "base_both_growth", "num_heads": 16, "num_hidden_features": 64, "desc": "base_both_16h_64d"},
]
base_settings = []
for arch in base_arch_conditions:
    for t_mode in transfer_modes:
        base_settings.append({
            "grow_condition": arch["name"],
            "num_heads": arch["num_heads"],
            "num_hidden_features": arch["num_hidden_features"],
            "transfer_mode": t_mode,
            "desc": f"{arch['desc']}_{t_mode}",
            "is_base_experiment": True,
        })

# 3. Base experiments with full re-initialization at the data switch (Tasks 63 to 71)
#    Same larger sizes as above, but nothing is carried over from before the switch.
base_reinit_settings = []
for arch in base_arch_conditions:
    base_reinit_settings.append({
        "grow_condition": arch["name"],
        "num_heads": arch["num_heads"],
        "num_hidden_features": arch["num_hidden_features"],
        "transfer_mode": "none",
        "desc": f"{arch['desc']}_none",
        "is_base_experiment": True,
    })

# 4. No-switch experiments (Tasks 72 to 107): the model is grown / re-initialized at epoch 50 exactly as in
#    the growth experiments, but the training sequences are NOT resampled, so the data stays the same.
#    "noswitch_base" keeps the 8h/32d size (no growth); the others grow to the target size.
noswitch_arch_conditions = [
    {"name": "noswitch_base", "num_heads": 8, "num_hidden_features": 32, "desc": "noswitch_8h_32d"},
    {"name": "noswitch_width_growth", "num_heads": 8, "num_hidden_features": 64, "desc": "noswitch_width_8h_64d"},
    {"name": "noswitch_head_growth", "num_heads": 16, "num_hidden_features": 32, "desc": "noswitch_head_16h_32d"},
]
noswitch_transfer_modes = ["attn_only", "mlp_only", "both", "none"]
noswitch_settings = []
for arch in noswitch_arch_conditions:
    for t_mode in noswitch_transfer_modes:
        noswitch_settings.append({
            "grow_condition": arch["name"],
            "num_heads": arch["num_heads"],
            "num_hidden_features": arch["num_hidden_features"],
            "transfer_mode": t_mode,
            "desc": f"{arch['desc']}_{t_mode}",
            "is_base_experiment": False,  # always starts from the small 8h/32d model
            "resample_at_switch": False,
        })

all_settings = grow_settings + base_settings + base_reinit_settings + noswitch_settings


def select_by_task_id(task_id):
    """Return (setting, seed_offset) for a SLURM array task id."""
    if task_id >= FIRST_NOSWITCH_TASK:
        offset_task = task_id - FIRST_NOSWITCH_TASK
        return noswitch_settings[offset_task % len(noswitch_settings)], offset_task // len(noswitch_settings)
    if task_id >= FIRST_BASE_TASK:
        offset_task = task_id - FIRST_BASE_TASK
        n_base_tasks = len(base_settings) * NUM_SEEDS
        if offset_task < n_base_tasks:
            return base_settings[offset_task % len(base_settings)], offset_task // len(base_settings)
        offset_task -= n_base_tasks
        return base_reinit_settings[offset_task % len(base_reinit_settings)], offset_task // len(base_reinit_settings)
    return grow_settings[task_id % len(grow_settings)], task_id // len(grow_settings)
