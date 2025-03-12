import onnx
import onnx_graphsurgeon as gs
import numpy as np

# ?? ONNX ??
model_path = "uniad_tiny_dummy.onnx"
onnx_model = onnx.load(model_path)

# ?? ONNX Graph Surgeon ?
graph = gs.import_onnx(onnx_model)

# ???? TopK ??
topk_nodes = [node for node in graph.nodes if node.op == "TopK"]

# ? TopK ??? K ????
for node in topk_nodes:
    k_input = node.inputs[1]
    k_value = k_input.values[0] if isinstance(k_input, gs.Constant) else 1  # ???? 1
    k_constant = gs.Constant(name=k_input.name, values=np.array([k_value], dtype=np.int64))
    node.inputs[1] = k_constant

# ???,???????????
graph.cleanup()

# ????????? ONNX ??
modified_model = gs.export_onnx(graph)

# ????????
modified_model_path = "uniad_tiny_dummy_modified.onnx"
onnx.save(modified_model, modified_model_path)
