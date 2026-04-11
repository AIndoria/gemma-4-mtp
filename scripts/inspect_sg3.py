import numpy as np
from gemma_mtp.tflite_loader import TFLiteModelReader

def inspect_sg3_constants():
    reader = TFLiteModelReader("data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite")
    # Subgraph 3
    sg = reader.model.Subgraphs(3)
    
    def get_const(idx):
        t = sg.Tensors(idx)
        buf_idx = t.Buffer()
        buf = reader.model.Buffers(buf_idx).DataAsNumpy()
        if buf is None: return None
        return np.frombuffer(buf.tobytes(), dtype=np.int32)

    # T 9, 10, 11
    print(f"T 9: {get_const(9)}")
    print(f"T 10: {get_const(10)}")
    print(f"T 11: {get_const(11)}")
    
    # Also check Op 0 slice indices
    # Op 0: SLICE [2, 12, 13] -> [14]
    print(f"T 12: {get_const(12)}")
    print(f"T 13: {get_const(13)}")

if __name__ == "__main__":
    inspect_sg3_constants()
