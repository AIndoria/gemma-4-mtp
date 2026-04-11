import numpy as np
from ai_edge_litert.interpreter import Interpreter
from gemma_mtp.tflite_loader import TFLiteModelReader

def probe_v_layout():
    model_path = "data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite"
    interpreter = Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    
    input_details = {detail["name"]: detail["index"] for detail in interpreter.get_input_details()}
    
    # First, get q_rope
    interpreter.set_tensor(input_details["mtp_drafter_activations:0"], np.ones((1, 1, 5120), dtype=np.float32))
    interpreter.set_tensor(input_details["mtp_drafter_input_pos:0"], np.array([0], dtype=np.int32))
    
    param_tensor = np.zeros((1, 1, 1, 7), dtype=np.int32)
    param_tensor[0, 0, 0, 0] = 0
    param_tensor[0, 0, 0, 1] = 512
    param_tensor[0, 0, 0, 2] = 512
    interpreter.set_tensor(input_details["mtp_drafter_param_tensor:0"], param_tensor)
    
    interpreter.invoke()
    
    q_rope = interpreter.get_tensor(77) # [1, 1, 4, 256]
    q_rope_val = q_rope[0, 0, :, :]
    
    # Set k_cache to match q_rope
    reader = TFLiteModelReader(model_path)
    k_info = reader.tensor_info("mtp_drafter_kv_cache_k_22:0")
    k_scale = k_info.quantization.scale[0]
    
    k_cache = np.zeros((1, 2, 32003, 256), dtype=np.int8)
    for h in range(2):
        k_cache[0, h, 0, :] = np.round(q_rope_val[h*2, :] / k_scale).astype(np.int8)
        k_cache[0, h, 1, :] = np.round(q_rope_val[h*2 + 1, :] / k_scale).astype(np.int8)
    interpreter.set_tensor(input_details["mtp_drafter_kv_cache_k_22:0"], k_cache)
    
    # Get o_proj weights
    o_proj_weights = reader.read_dequantized("layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general")
    v_scale = reader.tensor_info("mtp_drafter_kv_cache_v_22:0").quantization.scale[0]
    
    # Pass 1: set ONLY (h=0, q=0, d=5) to non-zero
    # This means v[h=0, d=5, k=0] = 40
    v = np.zeros((1, 2, 256, 32003), dtype=np.int8)
    v[0, 0, 5, 0] = 40
    interpreter.set_tensor(input_details["mtp_drafter_kv_cache_v_22:0"], v)
    interpreter.invoke()
    
    attn_out = interpreter.get_tensor(100)
    print(f"Attn Out non-zero count: {np.count_nonzero(attn_out)}")
    print(f"Attn Out first 10 non-zero: {attn_out[0, 0, np.nonzero(attn_out[0, 0, :])[0][:10]]}")
    
    # Check if any weight matches the first non-zero output
    first_nz_idx = np.nonzero(attn_out[0, 0, :])[0][0]
    val = attn_out[0, 0, first_nz_idx]
    print(f"Target value at index {first_nz_idx}: {val:.6f}")
    
    # Find weights that could produce this value
    # val = ctx_val * weight
    target_w = val / (40 * v_scale)
    print(f"Target weight: {target_w:.6f}")
    
    # Search in the first_nz_idx-th ROW of o_proj_weights
    matches_w = np.where(np.isclose(o_proj_weights[first_nz_idx, :], target_w, atol=1e-3))[0]
    print(f"Columns in row {first_nz_idx} matching target weight: {matches_w}")

    # Pass 2: set ONLY (h=0, q=1, d=5) to non-zero
    # This means v[h=0, d=5, k=1] = 40
    v = np.zeros((1, 2, 256, 32003), dtype=np.int8)
    v[0, 0, 5, 1] = 40
    interpreter.set_tensor(input_details["mtp_drafter_kv_cache_v_22:0"], v)
    interpreter.invoke()
    
    attn_out = interpreter.get_tensor(100)
    if np.count_nonzero(attn_out) > 0:
        matches = []
        for j in range(1024):
            if np.allclose(attn_out[0, 0, :], o_proj_weights[:, j] * (40 * v_scale), atol=1e-3):
                matches.append(j)
        print(f"(h=0, q=1, d=5) matches columns: {matches}")

if __name__ == "__main__":
    probe_v_layout()
