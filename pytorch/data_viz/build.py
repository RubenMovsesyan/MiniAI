"""Parse VOC annotations -> data.js manifest for the static viewer.

Reads object boxes only (parts ignored). Writes data.js next to index.html.
Run once (or whenever the dataset path changes):  python build.py
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

VOC_DIR = "/home/rubenmovsesyan/external_1/ml_datasets/pascal_voc/VOC2012"

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fff69f",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#4d79ff", "#a9a9a9",
]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


def build() -> None:
    ann_dir = Path(VOC_DIR) / "Annotations"
    images = []
    counts = [0] * len(CLASSES)
    for fn in sorted(os.listdir(ann_dir)):
        root = ET.parse(ann_dir / fn).getroot()
        size = root.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)
        objs, seen = [], set()
        for o in root.findall("object"):
            ci = CLS_IDX[o.find("name").text]
            b = o.find("bndbox")
            objs.append([ci, int(float(b.find("xmin").text)), int(float(b.find("ymin").text)),
                         int(float(b.find("xmax").text)), int(float(b.find("ymax").text))])
            seen.add(ci)
        for ci in seen:
            counts[ci] += 1
        images.append([fn[:-4], w, h, objs])

    out = Path(__file__).parent / "data.js"
    with out.open("w") as f:
        f.write(f"const VOC_DIR = {json.dumps(VOC_DIR)};\n")
        f.write(f"const CLASSES = {json.dumps(CLASSES)};\n")
        f.write(f"const COLORS = {json.dumps(COLORS)};\n")
        f.write(f"const COUNTS = {json.dumps(counts)};\n")
        f.write("const IMAGES = [\n")
        for im in images:
            f.write(json.dumps(im, separators=(",", ":")) + ",\n")
        f.write("];\n")
    print(f"wrote {out}  ({len(images)} images)")


if __name__ == "__main__":
    build()
