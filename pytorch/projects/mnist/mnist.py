"""End-to-end demo: load data, define a model, train, save best, visualize.

Run from inside pytorch/:  python -m projects.mnist.mnist
"""

import torch.nn as nn

from data.mnist import mnist_loaders
from training.trainer import Trainer

train, test = mnist_loaders(batch=100)

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10),
)

tr = Trainer(model, optimizer="adam", lr=1e-3, seed=42)
tr.fit(train, test, epochs=10, save_best="best.onnx")
tr.visualize()
tr.show_predictions(test)
