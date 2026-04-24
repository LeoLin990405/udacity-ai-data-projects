# Face Generation — DCGAN

Udacity **Deep Learning Nanodegree (nd101)** — Course 5: Building Generative Adversarial Networks, final project.

## Overview

Train a **Deep Convolutional GAN** from scratch on **CelebA** faces (64×64).

## Architecture

Standard DCGAN (Radford et al. 2015):

- **Generator**: z(100) → 4×4×512 → 8×8 → 16×16 → 32×32 → 64×64×3 (Tanh)
- **Discriminator**: 64×64×3 → 32×32 → 16×16 → 8×8 → 4×4 → logit (LeakyReLU + BatchNorm)
- **Loss**: BCE with logits
- **Optimizer**: Adam(lr=2e-4, β₁=0.5)
- **Init**: Normal(0, 0.02)

## Results (3 epochs × 5000 images, M5 MPS)

| Epoch | Avg lossG | Avg lossD | Time |
|-------|-----------|-----------|------|
| 1 | 15.73 | 0.34 | 130s |
| 2 | 12.08 | 0.86 | 109s |
| 3 | 5.55 | 0.83 | 124s |

Generated faces are recognizable (hair, skin tone, basic facial structure) — more epochs on the full 162K dataset sharpens features further.

## Stack

- Python 3.14 / PyTorch 2.10 / MPS backend
- HuggingFace `datasets` for CelebA loading (`tpremoli/CelebA-attrs`)

## Run

```bash
pip install torch torchvision datasets matplotlib
BATCH=64 EPOCHS=3 SUBSET=5000 python run_gan.py
```

## Files

- `project.ipynb` / `project.html` — submission notebook with loaded generator + sample images
- `run_gan.py` — full training script
- `samples/samples_ep0N.png` — face grids per epoch
- `loss_curve.png` — G vs D loss over training
- `checkpoints/` — generator.pt + discriminator.pt (ignored; regenerate with `run_gan.py`)

## Context

Part of Leo's [Udacity Masters in AI](https://github.com/LeoLin990405?tab=repositories) journey. See also:
- [landmark-classifier](https://github.com/LeoLin990405/landmark-classifier) — Course 3 CNN
- [text-translation-sentiment](https://github.com/LeoLin990405/text-translation-sentiment) — Course 4 Transformers
