"""Training + evaluation. This is the meat — reads like the old main().

    from train import train
    model = train(Config.from_env())
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data
from model import Config, Net


@torch.no_grad()
def evaluate(model: Net, loader: DataLoader, device: str) -> float:
    """Fraction of `loader` the model classifies correctly."""
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += yb.size(0)
    return correct / total


def train(cfg: Config) -> Net:
    print(f"config: {cfg}")
    train_loader, test_loader = data.make_loaders(cfg.data_dir, cfg.batch)

    # Put the model on GPU
    model = Net(cfg).to(cfg.device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    print("Starting Training...")
    t0 = time.perf_counter()
    best = 0.0
    for epoch in range(cfg.epochs):
        # Put the model in training mode, not important for this type of model but good habit for the future
        model.train()

        running_loss = 0.0
        for xb, yb in train_loader:
            # Move the batch of images and labels to the device
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            # Zero the gradients from the previous batch
            optimizer.zero_grad()

            # Pass the image batch through and get the output of the current model
            logits = model(xb)

            # Calculate the loss of this step (Cross entropy between model output and labels)
            loss = loss_function(logits, yb)

            # Compute the gradients from the loss function of the model
            loss.backward()

            # Update the models weights
            optimizer.step()

            # Keep track of loss to print it later
            running_loss += loss.item()

        # Loss + test accuracy for this epoch
        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, cfg.device)
        best = max(best, acc)
        print(f"Epoch [{epoch+1}/{cfg.epochs}] - loss: {avg_loss:.4f} - test acc: {acc:.2%}")

    secs = time.perf_counter() - t0
    final = evaluate(model, test_loader, cfg.device)
    print(f"Done in {secs:.1f}s - final test acc: {final:.2%} - best: {best:.2%}")
    return model
