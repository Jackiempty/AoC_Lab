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
