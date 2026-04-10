import numpy as np
from ai_edge_litert.interpreter import Interpreter

def probe_k_layout():
    model_path = "data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite"
    interpreter = Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    
    input_details = {detail["name"]: detail["index"] for detail in interpreter.get_input_details()}
    tensor_details = {detail["name"]: detail["index"] for detail in interpreter.get_tensor_details()}
    
    # Create unique sequence k
    # shape (1, 2, 32003, 256)
    k = np.zeros((1, 2, 32003, 256), dtype=np.int8)
    marker = np.array([123, 45, 67, 89, 101, -123, -45, -67], dtype=np.int8)
    k[0, 0, 500, :len(marker)] = marker
    k[0, 1, 600, :len(marker)] = marker + 1
    
    interpreter.set_tensor(input_details["mtp_drafter_kv_cache_k_22:0"], k)
    
    # ... dummy other inputs ...
    # (rest of set_tensor calls remain same)
    
    # Dummy other inputs
    interpreter.set_tensor(input_details["mtp_drafter_input_pos:0"], np.array([700], dtype=np.int32))
    interpreter.set_tensor(input_details["mtp_drafter_activations:0"], np.zeros((1, 1, 5120), dtype=np.float32))
    interpreter.set_tensor(input_details["mtp_drafter_param_tensor:0"], np.zeros((1, 1, 1, 7), dtype=np.int32))
    interpreter.set_tensor(input_details["mtp_drafter_mask:0"], np.ones((1, 1, 1, 32003), dtype=np.bool_))
    # caches
    for name in ["kv_cache_v_22", "kv_cache_v_23", "kv_cache_k_23"]:
        interpreter.set_tensor(input_details[f"mtp_drafter_{name}:0"], np.zeros(interpreter.get_input_details()[input_details[f"mtp_drafter_{name}:0"]]["shape"], dtype=np.int8))

    # Check if param_tensor was set correctly
    pt_idx = input_details["mtp_drafter_param_tensor:0"]
    pt_before = interpreter.get_tensor(pt_idx).copy()
    print(f"param_tensor before: {pt_before}")
    
    interpreter.invoke()
    
    pt_after = interpreter.get_tensor(pt_idx)
    print(f"param_tensor after: {pt_after}")
    
    # Now check internal tensors in layer 0
    # In SG 3, T 21 was the sliced k_cache [1, 2, 512, 256]
    # We can't see SG 3 tensors easily.
    # But maybe we can see them if they are preserved?
    # Usually subgraph tensors are named with the subgraph name prefix.
    
    def find_sequence(tensor, seq):
        if tensor.dtype == np.int8:
            # Simple sliding window search on flattened tensor
            flat = tensor.flatten()
            matches = []
            for i in range(len(flat) - len(seq) + 1):
                if np.array_equal(flat[i:i+len(seq)], seq):
                    matches.append(i)
            return matches
        elif tensor.dtype == np.float32:
            # Try dequantized marker
            scale = 0.00596147496253252
            seq_f = seq.astype(np.float32) * scale
            flat = tensor.flatten()
            matches = []
            for i in range(len(flat) - len(seq) + 1):
                if np.allclose(flat[i:i+len(seq)], seq_f, atol=1e-5):
                    matches.append(i)
            return matches
        return []

    print("\nNon-zero tensors:")
    for detail in interpreter.get_tensor_details():
        idx = detail["index"]
        name = detail["name"]
        try:
            t = interpreter.get_tensor(idx)
            if t is None: continue
            if np.count_nonzero(t) > 0:
                print(f"Tensor {idx} ({name}), shape {t.shape}, dtype {t.dtype}, non-zero count {np.count_nonzero(t)}")
                if idx == 84:
                    # Print values around where we expect them
                    # k marker was at index 500
                    # q was zeros, so q @ k.T should be zeros unless something else is happening
                    # But we saw 125964 non-zero scores! Why?
                    # Maybe activations weren't really zero?
                    print(f"  scores at [0,0,0,500:510]: {t[0,0,0,500:510]}")
                    print(f"  max/min scores: {t.max()}, {t.min()}")
        except:
            continue

if __name__ == "__main__":
    probe_k_layout()
