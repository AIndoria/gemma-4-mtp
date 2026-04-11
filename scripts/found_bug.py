import numpy as np
from gemma_mtp.tflite_loader import TFLiteModelReader

def inspect_score_buffer():
    reader = TFLiteModelReader("data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite")
    # T 25 in SG 3 or T 25 in main?
    # Wait, in dump_sg 3: T 25: arith.constant212 [1, 2, 2, 32003]
    # But dump_sg 0 might have it too.
    # Actually, constants are usually in the model's Buffers.
    
    # I'll just check tensor at index 25 in SG 3.
    # Or more easily, check all constants of shape [1, 2, 2, 32003].
    
    # Actually, look at dump_sg 0 output:
    # Op 30: STABLEHLO_COMPOSITE [78, 3, 4] -> [84]
    # In main SG, T 3 is k_cache, T 4 is param_tensor.
    # T 84 is scores.
    
    # Let's find what T 36 is in main SG.
    # T 36: arith.constant5 [] FLOAT32. We saw it is -1e30.
    
    # Let's look at Op 35: SELECT_V2 [88, 84, 36] -> [89]
    # This masks T 84 (scores) using T 88 (mask) and T 36 (-1e30).
    # This happens BEFORE softmax.
    
    # So if T 88 is only True for 189:700, then all other scores are -1e30.
    # Then softmax of 189:700 will be correct.
    
    # Now, context calculation:
    # Op 37: STABLEHLO_COMPOSITE [90, 1, 4] -> [91]
    # This must be doing the same slicing on v_cache (T 1) as it did on k_cache.
    
    # If it slices 189:700 from v_cache, and probs[..., 189:701] are the non-zero ones.
    # Then context = probs[..., 189:701] @ v_cache[..., 189:701].T.
    
    # But compare_internal_tflite_runtime.py does:
    # my_ctx = probs[..., 0:512] @ v_cache[..., 0:512].T
    
    # IF tokens are stored linearly at 0, 1, 2, ..., 32002.
    # THEN v_cache[..., 0:512] contains tokens 0:511.
    # BUT we need tokens 189:700.
    # So my_ctx uses the WRONG tokens!
    # Except tokens 189:511 which are in both.
    # For k in 512:700, they are in TFLite's window but NOT in 0:512.
    # For k in 0:188, they are in 0:512 but NOT in TFLite's window.
    
    # THAT IS WHY SIMILARITY IS LOW!

    print("Found it! The window in TFLite is SLIDING, but the PyTorch script uses FIXED 0:512.")

if __name__ == "__main__":
    inspect_score_buffer()
