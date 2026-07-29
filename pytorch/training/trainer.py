"""Trainer: fit any nn.Module, keep the best iteration, export it, visualize it."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.export import Exporter, OnnxExporter


def _optimizer(name_or_opt, params, lr):
    if not isinstance(name_or_opt, str):
        return name_or_opt
    if name_or_opt == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name_or_opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer '{name_or_opt}'")


class Trainer:
    def __init__(self, model: nn.Module, optimizer="adam", lr: float = 1e-3,
                 device: str = "cuda", exporter: Exporter = OnnxExporter(), seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = model.to(self.device)
        self.optimizer = _optimizer(optimizer, self.model.parameters(), lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.exporter = exporter
        self._example = None

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            correct += (self.model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
        return correct / total

    def fit(self, train: DataLoader, test: DataLoader, epochs: int,
            save_best: str | None = None, verbose: bool = True) -> float:
        if self._example is None:
            self._example = next(iter(train))[0][:1].to(self.device)

        best_acc, best_state = 0.0, None
        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            for xb, yb in train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                running += loss.item()
            acc = self.evaluate(test)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(self.model.state_dict())
            if verbose:
                print(f"epoch {epoch:2d}/{epochs}  loss {running / len(train):.4f}  test acc {acc:.2%}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose:
            print(f"done  best test acc {best_acc:.2%}")
        if save_best:
            self.export(save_best)
        return best_acc

    def export(self, path: str, example: torch.Tensor | None = None) -> str:
        ex = example if example is not None else self._example
        if ex is None:
            raise RuntimeError("no example input; fit first or pass example=")
        self.exporter.save(self.model, path, ex.to(self.device))
        return path

    def visualize(self, out: str = "network.html", input_shape=(1, 1, 28, 28)) -> str:
        from utils import netviz
        return netviz.render(self.model, out, input_shape)

    def show_predictions(self, loader: DataLoader) -> None:
        from utils import viewer
        viewer.show(self.model, loader, self.device)
