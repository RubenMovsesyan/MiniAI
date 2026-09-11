# artist — neural style transfer

Restyle a **content image** with the texture and palette of a **style image**, using a
frozen VGG16 as a perceptual feature extractor (Gatys, Ecker & Bethge 2015). Two inputs
in, one image out: the content's layout rendered in the style's brushwork.

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
.venv/bin/python -m projects.artist.artist path/to/content.jpg path/to/style.jpg out.png
```

## Files

| File | What |
|---|---|
| `artist.py` | entry point — `Config`, `StyleTransfer`, the optimisation loop |
| `vgg.py`    | `VGGFeatures` — frozen VGG16 conv stack, named activations, loads the submodule weights |
| `losses.py` | `gram_matrix`, `content_loss`, `style_loss`, `tv_loss` |
| `images.py` | load / save images, VGG mean-std normalisation round-trip |

Everything is a skeleton right now — signatures and structure only, bodies are `TODO`.
