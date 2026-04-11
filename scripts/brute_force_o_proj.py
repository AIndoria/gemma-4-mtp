import torch
import numpy as np
import itertools
from gemma_mtp.tflite_loader import TFLiteModelReader

def brute_force_o_proj():
    pos_val = 100
    outputs = np.load(f"data/derived/tflite_outputs_{pos_val}.npz")
    reader = TFLiteModelReader("data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite")
    
    # Context (T 91 output of attention context)
    ctx_f32 = outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"]
    # ctx_f32 shape is (1, 2, 2, 256) (B, H, Q, D)
    ctx_f32 = torch.from_numpy(ctx_f32)
    
    # Quantized input to o_proj (T 97)
    ctx_quant = outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite"]
    # ctx_quant shape is (1, 1, 1024) (B, T, 1024)
    ctx_quant = torch.from_numpy(ctx_quant)
    
    # Dequantized output of o_proj (T 100)
    target = outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite1"]
    # target shape is (1, 1, 256) (B, T, 256)
    target = torch.from_numpy(target)
    
    # Weights (T 98)
    w_raw = reader.read_raw(98) # (256, 1024) INT8
    w_info = reader.tensor_info(98)
    w_scales = torch.from_numpy(w_info.quantization.scale).view(-1, 1)
    w_f32 = torch.from_numpy(w_raw).float() * w_scales
    
    # Input scale (T 97)
    in_scale = reader.tensor_info(97).quantization.scale[0]
    
    # Output scale (T 100)
    # Wait, T 100 is dequantized. So target is already float.
    
    # target is already T 100 (composite1)
    # ctx_f32 is T 91
    
    for p in itertools.permutations(range(3)):
        # p is a permutation of (H, Q, D)
        ctx_p = ctx_f32.squeeze(0).permute(*p).reshape(-1)
        
        # Test if target is correlated with (ctx_p @ w_f32.T)
        out_p = (ctx_p @ w_f32.T)
        cosine = torch.nn.functional.cosine_similarity(out_p.flatten(), target.flatten(), dim=0)
        print(f"Permutation {p}: cosine {float(cosine):.6f}")

    print("\nTesting weight permutations...")
    # What if the 1024 dimension of weights is ordered differently?
    # w_f32 is (256, 1024)
    w_reshaped = w_f32.view(256, 2, 2, 256) # (out, H, Q, D)
    ctx_std = ctx_f32.reshape(1, 1, 1024) # (B, T, H*Q*D)
    
    for p, name in zip(perms, names):
        # Permute the 1024 dimensions of weights
        # p is a permutation of (H, Q, D) -> dimensions (1, 2, 3)
        w_p = w_reshaped.permute(0, *p).reshape(256, 1024)
        out_p = ctx_std @ w_p.T
        cosine = torch.nn.functional.cosine_similarity(out_p.flatten(), target.flatten(), dim=0)
        print(f"Weight permutation {name}: cosine {float(cosine):.6f}")

if __name__ == "__main__":
    brute_force_o_proj()
