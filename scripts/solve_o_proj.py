import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from gemma_mtp.tflite_loader import TFLiteModelReader

def solve_o_proj():
    model_path = "data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite"
    interpreter = Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    
    input_details = {detail["name"]: detail["index"] for detail in interpreter.get_input_details()}
    
    # We need to collect multiple (input, output) pairs for o_proj
    # input is T 97 (1024), output is T 99 (256) (quantized)
    # or input is T 96 (1024 f32), output is T 100 (256 f32)
    
    inputs_list = []
    outputs_list = []
    
    rng = np.random.default_rng(42)
    
    print("Collecting samples...")
    for i in range(10):
        # Set random activations and v_cache
        interpreter.set_tensor(input_details["mtp_drafter_activations:0"], rng.standard_normal((1, 1, 5120)).astype(np.float32))
        interpreter.set_tensor(input_details["mtp_drafter_kv_cache_v_22:0"], rng.integers(-10, 11, (1, 2, 256, 32003)).astype(np.int8))
        interpreter.set_tensor(input_details["mtp_drafter_kv_cache_k_22:0"], rng.integers(-10, 11, (1, 2, 32003, 256)).astype(np.int8))
        interpreter.set_tensor(input_details["mtp_drafter_input_pos:0"], np.array([100], dtype=np.int32))
        
        param_tensor = np.zeros((1, 1, 1, 7), dtype=np.int32)
        param_tensor[0, 0, 0, 0] = 0
        param_tensor[0, 0, 0, 1] = 512
        param_tensor[0, 0, 0, 2] = 512
        interpreter.set_tensor(input_details["mtp_drafter_param_tensor:0"], param_tensor)
        
        interpreter.invoke()
        
        # Collect T 96 (input to quantize before o_proj) and T 100 (dequantized output)
        inp = interpreter.get_tensor(96).flatten() # (1024)
        out = interpreter.get_tensor(100).flatten() # (256)
        
        inputs_list.append(inp)
        outputs_list.append(out)
        
    X = np.stack(inputs_list) # (10, 1024)
    Y = np.stack(outputs_list) # (10, 256)
    
    # We want to find W such that Y = X @ W.T
    # This is a system of equations for each output dimension i:
    # Y[:, i] = X @ W[i, :]
    
    # But wait! We have 1024 unknowns and only 10 samples.
    # This is underdetermined for one output dimension.
    # BUT we have 256 output dimensions. Still doesn't help.
    
    # Wait! I have ACHIEVED 1.0 similarity for context.
    # So I can just check which PERMUTATION of my_ctx matches the solved W.
    
    # I'll use 1024 samples!
    print("Collecting 1024 samples (this will be slow)...")
    # Actually, 1024 samples * 10 blocks... maybe too slow.
    # I'll use 128 samples and see if I can find a pattern.
    
    # Wait! I have a better way. 
    # Use ONE-HOT activations to isolate columns.
    # But activations is 5120 large. 
    # Better: set v_cache to one-hot!
    # I already tried that and it didn't match.
    
    # I'll just check if o_proj weights are interleaved by 128 AGAIN.
    # I'll use the brute_force_o_proj.py script but I'll add more permutations.
    
if __name__ == "__main__":
    solve_o_proj()
