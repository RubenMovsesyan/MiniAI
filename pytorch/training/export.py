"""Model export. Swap Exporter for other formats later; ONNX is the only one now."""

from __future__ import annotations

import torch
import torch.nn as nn


class Exporter:
    ext = ""

    def save(self, model: nn.Module, path: str, example: torch.Tensor) -> None:
        raise NotImplementedError


class OnnxExporter(Exporter):
    ext = ".onnx"

    def save(self, model: nn.Module, path: str, example: torch.Tensor) -> None:
        model.eval()
        torch.onnx.export(
            model, example, path,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
            dynamo=False,
        )
