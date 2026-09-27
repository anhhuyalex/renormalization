import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import numpy as np
from grow_utils import grow_onelayer_attention_weights
import gpt

def run_tests():
    print("=" * 70)
    print("RUNNING NETWORK GROWING UTILITIES TEST SUITE ON SMALL NETWORKS")
    print("=" * 70)
    
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    vocab_size = 4
    len_context = 16
    num_mlp_layers = 2
    
    # Base network parameters (small network)
    base_heads = 2
    base_hidden = 4  # hidden features per head
    base_mlp = base_heads * base_hidden  # 8
    
    print(f"Base network: H={base_heads}, d={base_hidden}, total_hidden={base_heads*base_hidden}, mlp_dim={base_mlp}")
    
    configs = [
        ("base (continuation/retrain)", base_heads, base_hidden),
        ("width_growth (only width)", base_heads, base_hidden * 2),     # H=2, d=8
        ("head_growth (only heads)", base_heads * 2, base_hidden),     # H=4, d=4
        ("both_growth (both width & heads)", base_heads * 2, base_hidden * 2), # H=4, d=8
    ]
    
    modes = ["both", "attn_only", "mlp_only"]
    
    total_passed = 0
    total_runs = len(configs) * len(modes)
    
    for config_name, t_heads, t_hidden in configs:
        t_mlp = t_heads * t_hidden
        print(f"\n>>> Testing Target Architecture: {config_name}")
        print(f"    Target config: H={t_heads}, d={t_hidden}, total_hidden={t_heads*t_hidden}, mlp_dim={t_mlp}")
        
        for mode in modes:
            print(f"    --- Mode: {mode} ---")
            
            # 1. Create base model with distinct known weights
            base_model = gpt.OneLayerAttention(
                len_context=len_context,
                num_heads=base_heads,
                num_hidden_features=base_hidden,
                vocab_size=vocab_size,
                num_mlp_layers=num_mlp_layers,
                num_hidden_features_mlp=base_mlp
            ).to(device)
            
            # Fill with distinct recognizable values
            with torch.no_grad():
                for p in base_model.parameters():
                    p.copy_(torch.randn_like(p) * 0.5 + 2.0)
            base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
            
            # 2. Create target model initialized with distinct values
            target_model = gpt.OneLayerAttention(
                len_context=len_context,
                num_heads=t_heads,
                num_hidden_features=t_hidden,
                vocab_size=vocab_size,
                num_mlp_layers=num_mlp_layers,
                num_hidden_features_mlp=t_mlp
            ).to(device)
            
            with torch.no_grad():
                for p in target_model.parameters():
                    p.copy_(torch.randn_like(p) * 0.1 - 5.0)
            target_init_sd = {k: v.clone() for k, v in target_model.state_dict().items()}
            
            # 3. Apply growth utility
            grown_model = grow_onelayer_attention_weights(
                base_model=base_model,
                target_model=target_model,
                base_heads=base_heads,
                base_hidden=base_hidden,
                target_heads=t_heads,
                target_hidden=t_hidden,
                transfer_mode=mode
            )
            grown_sd = grown_model.state_dict()
            
            # 4. Rigorous weight slice assertions
            transfer_attn = mode in ["both", "attn_only"]
            transfer_mlp = mode in ["both", "mlp_only"]
            
            # Embeddings: wte and wpe
            for h in range(base_heads):
                t_s = h * t_hidden
                b_s = h * base_hidden
                if transfer_attn:
                    assert torch.allclose(grown_sd["wte.weight"][:, t_s : t_s + base_hidden], base_sd["wte.weight"][:, b_s : b_s + base_hidden]), f"wte slice mismatch at head {h}"
                    assert torch.allclose(grown_sd["wpe.weight"][:, t_s : t_s + base_hidden], base_sd["wpe.weight"][:, b_s : b_s + base_hidden]), f"wpe slice mismatch at head {h}"
                else:
                    assert torch.allclose(grown_sd["wte.weight"], target_init_sd["wte.weight"]), "wte should not be transferred in mlp_only mode"
                    assert torch.allclose(grown_sd["wpe.weight"], target_init_sd["wpe.weight"]), "wpe should not be transferred in mlp_only mode"
                    
            # Layer norms
            for ln in ["ln_f", "pre_attn_ln"]:
                for param in ["weight", "bias"]:
                    key = f"{ln}.{param}"
                    if transfer_attn:
                        for h in range(base_heads):
                            t_s = h * t_hidden
                            b_s = h * base_hidden
                            assert torch.allclose(grown_sd[key][t_s : t_s + base_hidden], base_sd[key][b_s : b_s + base_hidden]), f"{key} slice mismatch at head {h}"
                    else:
                        assert torch.allclose(grown_sd[key], target_init_sd[key]), f"{key} should not be transferred in mlp_only mode"
            
            # Attention projection
            C_s = base_heads * base_hidden
            C_t = t_heads * t_hidden
            if transfer_attn:
                for qkv in range(3):
                    r_s_base = qkv * C_s
                    r_t_base = qkv * C_t
                    for h in range(base_heads):
                        r_s = r_s_base + h * base_hidden
                        r_t = r_t_base + h * t_hidden
                        assert torch.allclose(grown_sd["c_attn.bias"][r_t : r_t + base_hidden], base_sd["c_attn.bias"][r_s : r_s + base_hidden]), f"c_attn.bias mismatch at qkv={qkv}, head={h}"
                        for hp in range(base_heads):
                            c_s = hp * base_hidden
                            c_t = hp * t_hidden
                            assert torch.allclose(grown_sd["c_attn.weight"][r_t : r_t + base_hidden, c_t : c_t + base_hidden], base_sd["c_attn.weight"][r_s : r_s + base_hidden, c_s : c_s + base_hidden]), f"c_attn.weight submatrix mismatch at head pair ({h}, {hp})"
            else:
                assert torch.allclose(grown_sd["c_attn.weight"], target_init_sd["c_attn.weight"]), "c_attn.weight should not be transferred in mlp_only mode"
                assert torch.allclose(grown_sd["c_attn.bias"], target_init_sd["c_attn.bias"]), "c_attn.bias should not be transferred in mlp_only mode"

            # MLP Layers
            M_s = base_heads * base_hidden
            if transfer_mlp:
                assert torch.allclose(grown_sd["linear_modules.0.bias"][:M_s], base_sd["linear_modules.0.bias"][:M_s]), "linear_modules.0.bias mismatch"
                for hp in range(base_heads):
                    c_s = hp * base_hidden
                    c_t = hp * t_hidden
                    assert torch.allclose(grown_sd["linear_modules.0.weight"][:M_s, c_t : c_t + base_hidden], base_sd["linear_modules.0.weight"][:M_s, c_s : c_s + base_hidden]), f"linear_modules.0.weight submatrix mismatch at head {hp}"
                assert torch.allclose(grown_sd["linear_modules.1.weight"][:M_s, :M_s], base_sd["linear_modules.1.weight"][:M_s, :M_s]), "linear_modules.1.weight mismatch"
                assert torch.allclose(grown_sd["linear_modules.1.bias"][:M_s], base_sd["linear_modules.1.bias"][:M_s]), "linear_modules.1.bias mismatch"
                assert torch.allclose(grown_sd["lm_head.weight"][:, :M_s], base_sd["lm_head.weight"][:, :M_s]), "lm_head.weight mismatch"
                assert torch.allclose(grown_sd["lm_head.bias"], base_sd["lm_head.bias"]), "lm_head.bias mismatch"
            else:
                assert torch.allclose(grown_sd["linear_modules.0.weight"], target_init_sd["linear_modules.0.weight"]), "MLP should not be transferred in attn_only mode"
                assert torch.allclose(grown_sd["lm_head.weight"], target_init_sd["lm_head.weight"]), "lm_head should not be transferred in attn_only mode"

            # 5. Forward and Backward pass check
            batch_size = 4
            x = torch.randint(0, vocab_size, (batch_size, len_context), device=device)
            y = torch.randint(0, vocab_size, (batch_size, len_context), device=device)
            
            logits = grown_model(x, None)
            assert logits.shape == (batch_size, len_context, vocab_size), f"Logits shape mismatch: {logits.shape}"
            assert not torch.isnan(logits).any(), "NaN detected in logits"
            assert not torch.isinf(logits).any(), "Inf detected in logits"
            
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            
            for p_name, p in grown_model.named_parameters():
                if "pre_attn_ln" in p_name:
                    # In gpt.py OneLayerAttention, pre_attn_ln is defined in __init__ but commented out in get_qkv
                    continue
                assert p.grad is not None, f"Gradient missing for {p_name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {p_name}"
                
            opt = torch.optim.Adam(grown_model.parameters(), lr=1e-3)
            opt.step()
            opt.zero_grad()
            
            total_passed += 1
            print("      ✓ PASS: Weight slices exact, ablation isolated, forward & backward pass verified.")

    # 6. Mini 2-phase training simulation
    print("\n" + "=" * 70)
    print("RUNNING MINI 2-PHASE END-TO-END TRAINING SIMULATION")
    print("=" * 70)
    
    # Phase 1: Train base model for 10 steps
    base_sim = gpt.OneLayerAttention(
        len_context=len_context,
        num_heads=base_heads,
        num_hidden_features=base_hidden,
        vocab_size=vocab_size,
        num_mlp_layers=num_mlp_layers,
        num_hidden_features_mlp=base_mlp
    ).to(device)
    sim_opt = torch.optim.Adam(base_sim.parameters(), lr=1e-2)
    
    x_train = torch.randint(0, vocab_size, (8, len_context), device=device)
    y_train = torch.randint(0, vocab_size, (8, len_context), device=device)
    
    initial_loss = nn.functional.cross_entropy(base_sim(x_train, None).view(-1, vocab_size), y_train.view(-1)).item()
    for step in range(10):
        sim_opt.zero_grad()
        out = base_sim(x_train, None)
        l = nn.functional.cross_entropy(out.view(-1, vocab_size), y_train.view(-1))
        l.backward()
        sim_opt.step()
    trained_loss = l.item()
    print(f"Phase 1 (Base model training): Initial Loss={initial_loss:.4f} -> After 10 steps Loss={trained_loss:.4f}")
    assert trained_loss < initial_loss, "Base model did not optimize"
    
    # Phase 2: Grow model to both_growth (H=4, d=8) with both transfer mode and resample data
    grown_sim = gpt.OneLayerAttention(
        len_context=len_context,
        num_heads=4,
        num_hidden_features=8,
        vocab_size=vocab_size,
        num_mlp_layers=num_mlp_layers,
        num_hidden_features_mlp=32
    ).to(device)
    
    grown_sim = grow_onelayer_attention_weights(
        base_model=base_sim,
        target_model=grown_sim,
        base_heads=base_heads,
        base_hidden=base_hidden,
        target_heads=4,
        target_hidden=8,
        transfer_mode="both"
    )
    
    # Resample new sequences (Phase 2 data)
    x_new = torch.randint(0, vocab_size, (8, len_context), device=device)
    y_new = torch.randint(0, vocab_size, (8, len_context), device=device)
    grown_opt = torch.optim.Adam(grown_sim.parameters(), lr=1e-2)
    
    init_phase2_loss = nn.functional.cross_entropy(grown_sim(x_new, None).view(-1, vocab_size), y_new.view(-1)).item()
    for step in range(10):
        grown_opt.zero_grad()
        out = grown_sim(x_new, None)
        l = nn.functional.cross_entropy(out.view(-1, vocab_size), y_new.view(-1))
        l.backward()
        grown_opt.step()
    final_phase2_loss = l.item()
    print(f"Phase 2 (Grown model training on resampled data): Initial Loss={init_phase2_loss:.4f} -> After 10 steps Loss={final_phase2_loss:.4f}")
    assert final_phase2_loss < init_phase2_loss, "Grown model did not optimize on resampled data"
    
    print("\n" + "=" * 70)
    print(f"ALL {total_passed}/{total_runs} SETTINGS + END-TO-END TRAINING SIMULATION PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
