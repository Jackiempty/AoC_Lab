import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx
from onnx import shape_inference

project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from layer_info import (
    ShapeParam,
    Conv2DShapeParam,
    LinearShapeParam,
    MaxPool2DShapeParam,
)

from lib.models.vgg import VGG
import torch2onnx


def parse_pytorch(model: nn.Module, input_shape=(1, 3, 32, 32)) -> list[ShapeParam]:
    layers = []
    #! <<<========= Implement here =========>>>
    def hook_fn(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            N, C, H, W = inputs[0].shape
            M, _, R, S = module.weight.shape
            E, F = output.shape[2], output.shape[3]
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            layers.append(
                Conv2DShapeParam(N, H, W, R, S, E, F, C, M, stride, padding)
            )
        elif isinstance(module, nn.MaxPool2d):
            N, C, H, W = inputs[0].shape
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            layers.append(MaxPool2DShapeParam(N, kernel_size, stride))
        elif isinstance(module, nn.Linear):
            N, in_features = inputs[0].shape
            layers.append(LinearShapeParam(N, module.in_features, module.out_features))

    # register hook
    hooks = []
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook_fn))

    # forward pass
    dummy_input = torch.randn(*input_shape)
    model(dummy_input)

    # remove hooks
    for h in hooks:
        h.remove()
    
    return layers


def parse_onnx(model: onnx.ModelProto) -> list[ShapeParam]:
    layers = []
    #! <<<========= Implement here =========>>>
    inferred_model = shape_inference.infer_shapes(model)
    
    def get_tensor_shape(tensor_name: str):
        for value_info in inferred_model.graph.value_info:
            if value_info.name == tensor_name:
                return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)
        
        for input_info in inferred_model.graph.input:
            if input_info.name == tensor_name:
                return tuple(dim.dim_value for dim in input_info.type.tensor_type.shape.dim)
        
        for output_info in inferred_model.graph.output:
            if output_info.name == tensor_name:
                return tuple(dim.dim_value for dim in output_info.type.tensor_type.shape.dim)
        
        return None
    
    for node in inferred_model.graph.node:
        op_type = node.op_type

        input_shape = get_tensor_shape(node.input[0]) if node.input else None
        output_shape = get_tensor_shape(node.output[0]) if node.output else None
        
        if op_type == "Conv":
            attrs = {a.name: a for a in node.attribute}
            R, S = attrs["kernel_shape"].ints
            U = attrs["strides"].ints[0] if "strides" in attrs else 1
            P = attrs["pads"].ints[0] if "pads" in attrs else 0

            N, C, H, W = input_shape
            N2, M, E, F = output_shape

            layers.append(
                Conv2DShapeParam(N=N, H=H, W=W, R=R, S=S, E=E, F=F,
                                C=C, M=M, U=U, P=P)
            )

        elif op_type == "MaxPool":
            attrs = {a.name: a for a in node.attribute}
            kH, kW = attrs["kernel_shape"].ints
            stride = attrs["strides"].ints[0]

            N, C, H, W = input_shape
            layers.append(MaxPool2DShapeParam(N=N, kernel_size=kH, stride=stride))

        elif op_type in ["Gemm", "MatMul"]:
            N, in_features = input_shape
            N2, out_features = output_shape
            layers.append(LinearShapeParam(N=N, in_features=in_features, out_features=out_features))

    return layers


def compare_layers(answer, layers):
    if len(answer) != len(layers):
        print(
            f"Layer count mismatch: answer has {len(answer)}, but ONNX has {len(layers)}"
        )

    min_len = min(len(answer), len(layers))

    for i in range(min_len):
        ans_layer = vars(answer[i])
        layer = vars(layers[i])

        diffs = {
            k: (ans_layer[k], layer[k])
            for k in ans_layer
            if k in layer and ans_layer[k] != layer[k]
        }

        if diffs:
            print(f"Difference in layer {i + 1} ({type(answer[i]).__name__}):")
            for k, (ans_val, val) in diffs.items():
                print(f"  {k}: answer = {ans_val}, onnx = {val}")

    if len(answer) > len(layers):
        print(f"Extra layers in answer: {answer[len(layers) :]}")
    elif len(layers) > len(answer):
        print(f"Extra layers in yours: {layers[len(answer) :]}")


def run_tests() -> None:
    """Run tests on the network parser functions."""
    answer = [
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=64, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=64, M=192, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=192, M=384, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=384, M=256, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=256, M=256, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        LinearShapeParam(N=1, in_features=4096, out_features=256),
        LinearShapeParam(N=1, in_features=256, out_features=128),
        LinearShapeParam(N=1, in_features=128, out_features=10),
    ]

    # Test with the PyTorch model.
    model = VGG()
    layers_pth = parse_pytorch(model)

    # Define the input shape.
    dummy_input = torch.randn(1, 3, 32, 32)
    # Save the model to ONNX.
    torch2onnx.torch2onnx(model, "parser_onnx.onnx", dummy_input)
    # Load the ONNX model.
    model_onnx = onnx.load("parser_onnx.onnx")
    layers_onnx = parse_onnx(model_onnx)

    # Display results.
    print("PyTorch Network Parser:")
    if layers_pth == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_pth)

    print("ONNX Network Parser:")
    if layers_onnx == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_onnx)


if __name__ == "__main__":
    run_tests()
