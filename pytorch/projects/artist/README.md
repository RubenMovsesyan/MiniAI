# artist — neural style transfer

Restyle a **content image** with the texture and palette of one or more **style
images**, using a frozen VGG16 as a perceptual feature extractor (Gatys, Ecker &
Bethge 2015). The content's layout, rendered in the style's brushwork — or, with
several style images, their *common* style: each image's Gram matrices are
averaged into a single target before the optimiser ever runs, so the result is a
genuine blend rather than a copy of any one input.

## Weights

VGG16 ImageNet weights live in the `weights/vgg16.tv_in1k` git submodule
(<https://huggingface.co/timm/vgg16.tv_in1k>). It was added with LFS smudge skipped, so
the large tensor files are 130-byte pointers until you fetch them:

```bash
git submodule update --init pytorch/projects/artist/weights/vgg16.tv_in1k
cd pytorch/projects/artist/weights/vgg16.tv_in1k && git lfs pull --include model.safetensors
```

`model.safetensors` (~528 MB) is all that's needed. `pytorch_model.bin` is the same
weights in the older pickle format — a fallback if `safetensors` isn't installed.

## Run

```bash
cd pytorch
.venv/bin/python -m projects.artist.artist path/to/content.jpg path/to/style.jpg -o out.png
# multiple style images -> optimised toward their common style, not any one of them
.venv/bin/python -m projects.artist.artist path/to/content.jpg style1.jpg style2.jpg style3.jpg -o out.png
```

## Files

| File | What |
|---|---|
| `artist.py` | entry point — `Config`, `StyleTransfer`, the optimisation loop |
| `vgg.py`    | `VGGFeatures` — frozen VGG16 conv stack, named activations, loads the submodule weights |
| `losses.py` | `gram_matrix`, `content_loss`, `style_loss`, `tv_loss` |
| `images.py` | load / save images, VGG mean-std normalisation round-trip |
