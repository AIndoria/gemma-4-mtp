import numpy as np
from gemma_mtp.tflite_loader import TFLiteModelReader

def inspect_constants():
    reader = TFLiteModelReader("data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite")
    
    # T 41: arith.constant15 [1, 1, 32003] INT32
    # T 40: arith.constant14 [1, 1, 32003] INT32
    
    # I need to find their indices in the main subgraph.
    # From dump_sg 0:
    # T 41 is at index 41
    # T 40 is at index 40
    
    # T 1: mtp_drafter_kv_cache_v_22:0
    v_info = reader.tensor_info(1)
    if v_info.quantization:
        print(f"V Cache scales: {v_info.quantization.scale}")
        print(f"V Cache zero_points: {v_info.quantization.zero_point}")
        print(f"V Cache quantized_dimension: {v_info.quantization.quantized_dimension}")
    else:
        print("V Cache not quantized")

    # T 3: mtp_drafter_kv_cache_k_22:0
    k_info = reader.tensor_info(3)
    if k_info.quantization:
        print(f"K Cache scales: {k_info.quantization.scale}")
        print(f"K Cache zero_points: {k_info.quantization.zero_point}")
        print(f"K Cache quantized_dimension: {k_info.quantization.quantized_dimension}")
    
    # Also check T 38, T 36 (select values)
    # T 36: arith.constant5 [] FLOAT32
    # T 38: arith.constant7 [] FLOAT32
    t36 = reader.read_raw(36)
    t38 = reader.read_raw(38)
    print(f"T 36: {t36}")
    print(f"T 38: {t38}")

if __name__ == "__main__":
    inspect_constants()
