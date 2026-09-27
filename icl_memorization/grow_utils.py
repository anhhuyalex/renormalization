import torch

def grow_onelayer_attention_weights(base_model, target_model, base_heads, base_hidden, target_heads, target_hidden, transfer_mode="both"):
    """
    Transfers weights from a base OneLayerAttention model to a target OneLayerAttention model.
    
    Args:
        base_model: Source model or state_dict (H_base, d_base)
        target_model: Target model (H_target, d_target)
        base_heads: Number of attention heads in base model
        base_hidden: Hidden features per head in base model
        target_heads: Number of attention heads in target model
        target_hidden: Hidden features per head in target model
        transfer_mode: Transfer ablation mode:
            - 'both': transfer both attention & MLP weights
            - 'attn_only': transfer attention & embeddings, keep target MLP random init
            - 'mlp_only': transfer MLP & LM head, keep target attention random init
            
    Returns:
        target_model with transferred weights loaded.
    """
    base_sd = base_model.state_dict() if hasattr(base_model, "state_dict") else base_model
    target_sd = target_model.state_dict()
    
    transfer_attn = transfer_mode in ["both", "attn_only"]
    transfer_mlp = transfer_mode in ["both", "mlp_only"]
    
    if transfer_attn:
        # 1. Embeddings: wte and wpe
        for h in range(base_heads):
            t_start = h * target_hidden
            s_start = h * base_hidden
            if "wte.weight" in target_sd and "wte.weight" in base_sd:
                target_sd["wte.weight"][:, t_start : t_start + base_hidden] = base_sd["wte.weight"][:, s_start : s_start + base_hidden]
            if "wpe.weight" in target_sd and "wpe.weight" in base_sd:
                target_sd["wpe.weight"][:, t_start : t_start + base_hidden] = base_sd["wpe.weight"][:, s_start : s_start + base_hidden]
        
        # 2. LayerNorms: ln_f and pre_attn_ln
        for ln_name in ["ln_f", "pre_attn_ln"]:
            for param in ["weight", "bias"]:
                key = f"{ln_name}.{param}"
                if key in target_sd and key in base_sd:
                    for h in range(base_heads):
                        t_start = h * target_hidden
                        s_start = h * base_hidden
                        target_sd[key][t_start : t_start + base_hidden] = base_sd[key][s_start : s_start + base_hidden]
        
        # 3. Attention projection: c_attn
        C_s = base_heads * base_hidden
        C_t = target_heads * target_hidden
        for qkv_idx in range(3):
            qkv_s_offset = qkv_idx * C_s
            qkv_t_offset = qkv_idx * C_t
            for h in range(base_heads):
                r_s = qkv_s_offset + h * base_hidden
                r_t = qkv_t_offset + h * target_hidden
                if "c_attn.bias" in target_sd and "c_attn.bias" in base_sd:
                    target_sd["c_attn.bias"][r_t : r_t + base_hidden] = base_sd["c_attn.bias"][r_s : r_s + base_hidden]
                for hp in range(base_heads):
                    c_s = hp * base_hidden
                    c_t = hp * target_hidden
                    target_sd["c_attn.weight"][r_t : r_t + base_hidden, c_t : c_t + base_hidden] =                         base_sd["c_attn.weight"][r_s : r_s + base_hidden, c_s : c_s + base_hidden]
                        
    if transfer_mlp:
        # 4. First MLP layer: linear_modules.0 (maps C -> M)
        M_s = base_heads * base_hidden
        if "linear_modules.0.bias" in target_sd and "linear_modules.0.bias" in base_sd:
            target_sd["linear_modules.0.bias"][:M_s] = base_sd["linear_modules.0.bias"][:M_s]
        for hp in range(base_heads):
            c_s = hp * base_hidden
            c_t = hp * target_hidden
            target_sd["linear_modules.0.weight"][:M_s, c_t : c_t + base_hidden] =                 base_sd["linear_modules.0.weight"][:M_s, c_s : c_s + base_hidden]
        
        # 5. Subsequent MLP layers: linear_modules.1, etc. (maps M -> M)
        mlp_idx = 1
        while f"linear_modules.{mlp_idx}.weight" in target_sd and f"linear_modules.{mlp_idx}.weight" in base_sd:
            target_sd[f"linear_modules.{mlp_idx}.weight"][:M_s, :M_s] = base_sd[f"linear_modules.{mlp_idx}.weight"][:M_s, :M_s]
            if f"linear_modules.{mlp_idx}.bias" in target_sd and f"linear_modules.{mlp_idx}.bias" in base_sd:
                target_sd[f"linear_modules.{mlp_idx}.bias"][:M_s] = base_sd[f"linear_modules.{mlp_idx}.bias"][:M_s]
            mlp_idx += 1
            
        # 6. LM Head: lm_head (maps M -> vocab_size)
        if "lm_head.weight" in target_sd and "lm_head.weight" in base_sd:
            target_sd["lm_head.weight"][:, :M_s] = base_sd["lm_head.weight"][:, :M_s]
        if "lm_head.bias" in target_sd and "lm_head.bias" in base_sd:
            target_sd["lm_head.bias"][:] = base_sd["lm_head.bias"][:]
            
    target_model.load_state_dict(target_sd)
    return target_model
