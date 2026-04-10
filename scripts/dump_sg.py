import tflite
from gemma_mtp.tflite_loader import TFLiteModelReader

def dump_subgraph(sg_idx):
    reader = TFLiteModelReader("data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite")
    sg = reader.model.Subgraphs(sg_idx)
    
    opcode_to_name = {getattr(tflite.BuiltinOperator, name): name for name in dir(tflite.BuiltinOperator) if not name.startswith("__")}
    tensor_type_to_name = {getattr(tflite.TensorType, name): name for name in dir(tflite.TensorType) if not name.startswith("__")}
    
    print(f"--- Subgraph {sg_idx} ---")
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        opcode = reader.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        opname = opcode_to_name.get(opcode, str(opcode))
        
        inputs = [int(op.Inputs(j)) for j in range(op.InputsLength())]
        outputs = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
        
        print(f"Op {i}: {opname} {inputs} -> {outputs}")
        
    print("\nTensors:")
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        shape = [int(t.Shape(j)) for j in range(t.ShapeLength())]
        print(f"T {i}: {t.Name().decode()} {shape} {tensor_type_to_name.get(t.Type(), str(t.Type()))}")

if __name__ == "__main__":
    import sys
    sg_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dump_subgraph(sg_idx)
