import torch
import numpy as np
import itertools

def try_permutations():
    # Load dumped tensors
    outputs = np.load("data/derived/tflite_internal_compare_outputs.npz")
    inputs = np.load("data/derived/tflite_compare_inputs.npz")
    
    # Target context
    target = torch.from_numpy(outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"])
    print("Target shape:", target.shape)
    
    # Probabilities
    scores_raw = torch.from_numpy(outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite"])
    
    # PyTorch q and k
    q_torch = torch.from_numpy(outputs["layer_0/layer_0.pre_q/attn.pre_q/reshape"])
    k_raw = torch.from_numpy(inputs["kv_cache_k_22"])
    k_scale = 0.00596147496253252
    k = k_raw.float() * k_scale
    
    # Compute scores: q @ k.T
    # q: (1, 2, 2, 256), k: (1, 2, 32003, 256)
    
    # Apply RoPE to k at each position
    head_dim = 256
    half_dim = head_dim // 2
    rope_base = 10000.0
    
    indices = torch.arange(32003).float()
    exponent = torch.arange(half_dim).float() / half_dim
    inv_freq = torch.pow(torch.tensor(rope_base), -exponent)
    
    # (T, half_dim)
    angles = indices.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).view(1, 1, 32003, half_dim)
    sin = torch.sin(angles).view(1, 1, 32003, half_dim)
    
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    k_rope = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    
    my_scores_k_rope = torch.einsum("bhqd,bhkd->bhqk", q_torch, k_rope)
    print(f"q @ k_rope.T cosine: {torch.nn.functional.cosine_similarity(my_scores_k_rope.flatten(), scores_raw.flatten(), dim=0).item()}")
    
    # Try q without RoPE @ k without RoPE?
    # Wait, q matched with RoPE already.
    
    # What if k is RoPE-ed at the SAME position as q? (input_pos = 700)
    indices_700 = torch.full((32003,), 700.0)
    angles_700 = indices_700.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos_700 = torch.cos(angles_700).view(1, 1, 32003, half_dim)
    sin_700 = torch.sin(angles_700).view(1, 1, 32003, half_dim)
    k_rope_700 = torch.cat([k1 * cos_700 - k2 * sin_700, k1 * sin_700 + k2 * cos_700], dim=-1)
    
    my_scores_k_rope_700 = torch.einsum("bhqd,bhkd->bhqk", q_torch, k_rope_700)
    print(f"q @ k_rope_700.T cosine: {torch.nn.functional.cosine_similarity(my_scores_k_rope_700.flatten(), scores_raw.flatten(), dim=0).item()}")

    # Try RMSNorm on k
    def rms_norm(x, eps=1e-6):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + eps)
    
    k_win = k[..., 189:701, :]
    scores_win = scores_raw[..., 189:701]
    # Let's see if k_win matches anything in TFLite?
    # We don't have k_win from TFLite easily.
    
    # Wait! In SG 3, T 22 was dequantized sliced k_cache [1, 2, 512, 256].
    # And we know TFLite is using our q.
    
    # What if the scale is wrong?
    # ref = a * my_score + b
    # 0.6823 * my_score + 0.2051
    
    # If ref = q @ (k * scale_adj).T
    # then scale_adj should be 0.6823?
    
    print("\nTrying adjusted k scale...")
    k_adj = k_win * 0.6823
    my_s_adj = torch.einsum("bhqd,bhkd->bhqk", q_torch, k_adj)
    sim_adj = torch.nn.functional.cosine_similarity(my_s_adj.flatten(), scores_win.flatten(), dim=0)
    print(f"k * 0.6823 cosine: {sim_adj.item()}")
    
    # Cosine similarity is invariant to scale! 
    # So changing scale won't help cosine similarity.
    # Only a different RELATIVE distribution of values helps.
    
    # This means q @ k.T is NOT what TFLite is doing, 
    # OR my k is not what TFLite is using.
    
    print("\nRegression test (offset 0):")
    q0 = q_torch[0, 0, 0, :] # first head, first query
    k0 = k_win[0, 0, :, :] # first head, all keys in window
    s0_ref = scores_win[0, 0, 0, :]
    s0_my = torch.matmul(q0, k0.T)
    
    # Try scaling
    s0_scaled = s0_my / (256 ** 0.5)
    print(f"My score mean: {s0_my.mean().item():.4f}, ref mean: {s0_ref.mean().item():.4f}")
    print(f"My scaled score mean: {s0_scaled.mean().item():.4f}")
    
    # Linear fit manually since sklearn is missing
    # y = a*x + b
    # x = s0_my, y = s0_ref
    x = s0_my
    y = s0_ref
    x_mean = x.mean()
    y_mean = y.mean()
    a = torch.sum((x - x_mean) * (y - y_mean)) / torch.sum((x - x_mean)**2)
    b = y_mean - a * x_mean
    
    y_pred = a * x + b
    r2 = 1 - torch.sum((y - y_pred)**2) / torch.sum((y - y_mean)**2)
    print(f"Linear fit: ref = {a.item():.4f} * my_score + {b.item():.4f}, R^2 = {r2.item():.4f}")
    
    # What if we use 1/sqrt(D) as a?
    a_inv_sqrt = 256**-0.5
    y_pred_inv_sqrt = a_inv_sqrt * x
    r2_inv_sqrt = 1 - torch.sum((y - y_pred_inv_sqrt)**2) / torch.sum((y - y_mean)**2)
    print(f"Fixed scale (1/sqrt(256)) R^2: {r2_inv_sqrt.item():.4f}")
    
    print("\nTrying more permutations...")
    # k is (1, 2, 32003, 256) originally (B, H, T, D)
    
    # 1. (B, T, H, D) -> (1, 32003, 2, 256)
    k_bthd = k.permute(0, 2, 1, 3)
    k_bthd_win = k_bthd[:, 189:701, :, :]
    my_s = torch.einsum("bhqd,bthd->bhqt", q_torch, k_bthd_win)
    print(f"k (B,T,H,D) cosine: {torch.nn.functional.cosine_similarity(my_s.flatten(), scores_win.flatten(), dim=0).item()}")

    # 2. (B, H, D, T) -> (1, 2, 256, 32003)
    k_bhdt = k.permute(0, 1, 3, 2)
    k_bhdt_win = k_bhdt[..., 189:701]
    my_s = torch.einsum("bhqd,bhdt->bhqt", q_torch, k_bhdt_win)
    print(f"k (B,H,D,T) cosine: {torch.nn.functional.cosine_similarity(my_s.flatten(), scores_win.flatten(), dim=0).item()}")

    # 3. (B, T, D, H) -> (1, 32003, 256, 2)
    k_btdh = k.permute(0, 2, 3, 1)
    k_btdh_win = k_btdh[:, 189:701, :, :]
    my_s = torch.einsum("bhqd,btdh->bhqt", q_torch, k_btdh_win)
    print(f"k (B,T,D,H) cosine: {torch.nn.functional.cosine_similarity(my_s.flatten(), scores_win.flatten(), dim=0).item()}")

    # What if V cache is (B, H, T, D) but was reported as (B, H, D, T)?
    # No, that's unlikely.
    
    # Try different synthetic k layouts
    # When we build synthetic inputs, we use rng.integers(..., size=shape)
    # shape is (1, 2, 32003, 256)
    # TFLite input detail for k_22 is (1, 2, 32003, 256)
    
    # What if the 2 heads are actually different in our synthetic input and we can tell?
    # Actually, they are random.
    
    # What if TFLite expects (B, T, H, D) but we pass (B, H, T, D)?
    # input_pos is 700.
    # If TFLite thinks it's (B, T, H, D), then when it slices T=512 window,
    # it's slicing the H dimension if we passed (B, H, T, D)!
    
    print("\nChecking if T/H are swapped in TFLite interpretation of input...")
    # our k: [1, 2, 32003, 256] (B, H, T, D)
    # If TFLite thinks it is [1, 32003, 2, 256] (B, T, H, D)
    # and it slices T [189:701], it is slicing our T dimension if it knows it's the 2nd dim.
    # But wait, TFLite input detail says [1, 2, 32003, 256]. So it knows H=2, T=32003.
    
    # What if q is grouped differently?
    # q is (1, 2, 2, 256) -> B, H_kv, Q_per_kv, D
    # TFLite's reshape op (Op 24) produces [1, 2, 2, 256] from [1, 1, 4, 256]
    # My reshape_grouped_query:
    #   q = q.view(batch_size, steps, spec.query_heads, spec.query_head_dim)
    #   q = q.view(batch_size, steps, spec.kv_heads, spec.queries_per_kv, spec.query_head_dim)
    #   return q.permute(0, 2, 1, 3, 4).reshape(batch_size, spec.kv_heads, steps * spec.queries_per_kv, spec.query_head_dim)
    # For steps=1: (1, 2, 2, 256).
    
    # Let's check TFLite's reshape logic for q.
    # Op 24: RESHAPE [77, 42] -> [78]
    # 77: [1, 1, 4, 256], 42: [4]

    print("\nChecking if param_tensor indexes k...")
    # param_tensor: [[[[54 87 75 41 97 84 49]]]] (random-ish)
    # TFLite input detail for param_tensor is (1, 1, 1, 7).
    pt = inputs["param_tensor"].flatten()
    print(f"param_tensor values: {pt}")

    # What if the window is [base : base+512]?
    # Where does base come from?
    # base = input_pos - window_size + 1?
    # input_pos = 700. 700 - 512 + 1 = 189.

    # Let's try different input_pos logic
    # In some runtimes, input_pos is the NEXT token position.
    # So if we want to attend to tokens 0..699, input_pos=700.

    # Wait, 0.80 is high but not 1.0. 
    # If the window was completely different, similarity would be 0.
    # 0.80 means we are MOSTLY right but SOME values are different.

    # What if only SOME queries in the grouped query are right?
    # grouped_query: (1, 2, 2, 256) -> B, H, Q, D
    for h in range(2):
        for q in range(2):
            my_s_hq = torch.matmul(q_torch[0, h, q, :], k_win[0, h, :, :].T)
            ref_s_hq = scores_win[0, h, q, :]
            sim = torch.nn.functional.cosine_similarity(my_s_hq.flatten(), ref_s_hq.flatten(), dim=0)
            print(f"H{h} Q{q} similarity: {sim.item()}")


if __name__ == "__main__":
    try_permutations()
