"""DetectionLoss: box + objectness + class loss over YOLO's flattened multi-scale
predictions, [B, 4+1+num_classes, num_anchors] (box, obj, cls channel order — see
YOLO.forward / flatten_and_concat in yolo.py). Anchor-to-ground-truth target assignment
isn't built yet (needs the VOC label loader); until then, targets must already be
provided in this same shape."""

import torch
import torch.nn as nn

from projects.YOLO.yolo import NUM_CLASSES


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 box_weight: float = 1.0, obj_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight, self.obj_weight, self.cls_weight = box_weight, obj_weight, cls_weight
        self.box_loss = nn.L1Loss()
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        box_p, obj_p, cls_p = preds.split([4, 1, self.num_classes], dim=1)
        box_t, obj_t, cls_t = targets.split([4, 1, self.num_classes], dim=1)
        return (self.box_weight * self.box_loss(box_p, box_t)
                + self.obj_weight * self.obj_loss(obj_p, obj_t)
                + self.cls_weight * self.cls_loss(cls_p, cls_t))


if __name__ == "__main__":
    preds = torch.randn(2, 4 + 1 + NUM_CLASSES, 340, requires_grad=True)
    targets = torch.rand(2, 4 + 1 + NUM_CLASSES, 340)
    loss = DetectionLoss()(preds, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert preds.grad is not None
    print("ok")
