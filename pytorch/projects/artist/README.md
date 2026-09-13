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
| `losses.py` | `gram_matrix`, `content_loss`, `style_loss`, `tv_loss`, `color_loss`, `color_tv_loss` |
| `images.py` | load / save images, VGG mean-std normalisation round-trip, YCbCr/`preserve_color` |

## Known issue: "fingerprint"/maze texture in flat content regions

With a heavy shallow-layer style reweight (e.g. `conv1_1=4.0, conv2_1=3.0`),
regions of the content with little to no *directionally coherent* local
gradient (e.g. Mona Lisa's chest/neck skin) render as a swirling maze/
fingerprint pattern instead of coherent brush strokes -- confirmed present
even with `color_weight=0`, so it predates and is unrelated to the color_loss
work. Root cause (measured with a structure-tensor coherence check, see
git history around 2026-09-12/13 for the diagnostic script): the chest has
comparable or higher raw edge energy than regions that stylize well (sleeve,
hair), but much lower orientation *coherence* -- its local gradients point in
inconsistent directions pixel-to-pixel, while sleeve/hair gradients stay
aligned along the real fold lines/strands. `style_loss`'s Gram matrices only
constrain the aggregate mixture of edge-orientation responses over the whole
image, never which direction any one patch should point -- regions with a
coherent content gradient get that direction "for free" as a tie-breaker;
regions without one have nothing to align adjacent patches to, so they settle
into a labyrinthine pattern instead.

Options identified (in rough order of effort):

1. **TRIED, REJECTED: spatially-varying style weight** using a content-
   coherence mask -- see
   `images/mona_lisa/network_tweaking/van_gogh_mona_lisa_coherence_mask.txt`.
   Implemented `losses.coherence_map`/`gram_matrix_masked`/`style_loss_masked`
   and `artist.Config.coherence_mask_strength`/`coherence_blur_size`/
   `coherence_threshold`. Three variants tried (raw mask, blurred mask, then
   a binarised threshold) -- all either worse than the original or, at best,
   a wash. This is now understood to be a structural ceiling, not a tuning
   problem: masking only controls how much undirected Gram-matching pressure
   a region gets, from none (smooth/photographic) to full (a maze) -- it
   never supplies an actual DIRECTION for a stroke to follow, so it can't
   produce real brush strokes in the chest at any setting.
2. **TRIED, REJECTED: add a shallower layer to `content_layers`**
   (`conv2_2`, alongside the existing `conv4_2`) -- see
   `images/mona_lisa/network_tweaking/van_gogh_mona_lisa_conv2_2.txt`.
   Made the fingerprint texture MORE pronounced, not less, while the overall
   colour palette was unaffected (this option was never about colour).
   `color_loss`/`color_weight` were also removed from `artist.py` entirely
   afterward (functions kept in `losses.py`) since combining the two gave
   "worst of both worlds" -- still fully colour-shifted and a worse texture.
3. **TRIED, REJECTED (and backwards from predicted): reduce the shallow-layer
   style reweight** -- see
   `images/mona_lisa/network_tweaking/van_gogh_mona_lisa_slw_sweep.txt`.
   Swept half/quarter/equal weighting: the fingerprint pattern got WORSE
   (denser, more uniform) at every step, not better, while overall
   composition and stroke scale stayed basically unchanged. Revised
   understanding: Gram matrices are position-blind at every layer, shallow
   or deep, so shifting weight toward deeper layers doesn't remove the root
   cause (no coherent content gradient to anchor a stroke to in that
   region) -- it just changes which layer's texture ends up filling the
   void, and the deeper-layer version was worse here.
4. **TRIED, REJECTED: plain `tv_weight`** (already in `Config`, currently
   0.0 everywhere) -- see
   `images/mona_lisa/network_tweaking/van_gogh_mona_lisa_tv_sweep.txt`.
   Swept 1/5/20/50: the fingerprint pattern barely changed even at 50, while
   real brush-stroke sharpness elsewhere (sleeve, robe) was already visibly
   degrading. A general isotropic smoothness prior can't tell "good" texture
   from "bad" -- it penalises both equally, so it flattens everything before
   it fixes anything.
5. **NOT the discarded orientation-loss idea** (see `images/starship/
   network_tweaking/`) -- its target vector's magnitude reflects directional
   confidence, so in a genuinely incoherent region it would likely produce a
   near-zero target too, i.e. it probably wouldn't impose a direction on the
   chest either. Noted so this isn't re-tried expecting a different result.

All five tried-or-considered options are now closed out, with the same
underlying lesson: everything on this list either changes the loss
*everywhere* (2, 4), changes which existing term dominates (3), or changes
how *much* undirected pressure a region gets (1) -- none of them ever give
the optimiser an actual direction to align a stroke to in a region the
content itself has none. A real fix would need to supply one, e.g. by
diffusing/extending a nearby coherent edge (collar, hairline) into the
incoherent interior -- a materially bigger task than any of the config-level
knobs tried here. Until/unless that's attempted, this artifact is a known
limitation of this style/content pairing at this config, not a bug to keep
chasing with more weight-tuning.
