"""v2 — Iterative improvement of Face Generation DCGAN.

Motivation (from v1 reviewer feedback):
- v1 baseline (BCE, Adam lr=2e-4, β₁=0.5, 3 epochs, 5000 images) → generated faces too blurry
- Reviewer rubric requires iteration: observe weak results → adjust hyperparameters → retrain → compare

Changes in v2:
 1. Larger dataset:         5000 → 10000 images (2x)
 2. More epochs:            3 → 15 (5x, gives G more time to learn structure)
 3. Lower learning rate:    2e-4 → 1e-4 (halved — reduces G/D oscillation, more stable convergence)
 4. Added data augmentation: RandomHorizontalFlip(p=0.5)
 5. Added beta2=0.999 explicitly (stability)
 6. Save samples every epoch → can show progression
"""
import gc
import os
import sys
import time


def main():
    sys.stdout.reconfigure(line_buffering=True)

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] device={device}", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples_v2", exist_ok=True)

    IMG_SIZE = 64
    Z_DIM = 100
    G_CHANNELS = 64
    D_CHANNELS = 64
    BATCH = int(os.environ.get("BATCH", "64"))
    EPOCHS = int(os.environ.get("EPOCHS", "15"))
    SUBSET = int(os.environ.get("SUBSET", "10000"))
    LR = float(os.environ.get("LR", "1e-4"))
    BETA1 = 0.5

    print(f"[{time.strftime('%H:%M:%S')}] cfg: BATCH={BATCH} EP={EPOCHS} SUBSET={SUBSET} LR={LR}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading CelebA subset from HF...", flush=True)
    hf = load_dataset("tpremoli/CelebA-attrs", split=f"train[:{SUBSET}]")

    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.RandomHorizontalFlip(p=0.5),  # v2: augmentation
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    class CelebADS(Dataset):
        def __init__(self, hf_ds):
            self.ds = hf_ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img = self.ds[idx]["image"].convert("RGB")
            return transform(img)

    ds = CelebADS(hf)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    print(f"[{time.strftime('%H:%M:%S')}] dataset: {len(ds)} images, {len(loader)} batches/epoch", flush=True)

    class Generator(nn.Module):
        def __init__(self, z_dim=Z_DIM, g_ch=G_CHANNELS):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, g_ch * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(g_ch * 8), nn.ReLU(True),
                nn.ConvTranspose2d(g_ch * 8, g_ch * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g_ch * 4), nn.ReLU(True),
                nn.ConvTranspose2d(g_ch * 4, g_ch * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g_ch * 2), nn.ReLU(True),
                nn.ConvTranspose2d(g_ch * 2, g_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(g_ch), nn.ReLU(True),
                nn.ConvTranspose2d(g_ch, 3, 4, 2, 1, bias=False),
                nn.Tanh(),
            )

        def forward(self, z):
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self, d_ch=D_CHANNELS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, d_ch, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(d_ch, d_ch * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d_ch * 2), nn.LeakyReLU(0.2, True),
                nn.Conv2d(d_ch * 2, d_ch * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d_ch * 4), nn.LeakyReLU(0.2, True),
                nn.Conv2d(d_ch * 4, d_ch * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d_ch * 8), nn.LeakyReLU(0.2, True),
                nn.Conv2d(d_ch * 8, 1, 4, 1, 0, bias=False),
            )

        def forward(self, x):
            return self.net(x).view(-1)

    def weights_init(m):
        cn = m.__class__.__name__
        if "Conv" in cn:
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif "BatchNorm" in cn:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, Z_DIM, 1, 1, device=device)

    g_losses, d_losses = [], []

    print(f"[{time.strftime('%H:%M:%S')}] G params: {sum(p.numel() for p in G.parameters()):,}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] training start", flush=True)

    for epoch in range(1, EPOCHS + 1):
        ep_g, ep_d = 0.0, 0.0
        t0 = time.time()
        for i, real in enumerate(loader):
            real = real.to(device)
            bsz = real.size(0)

            D.zero_grad()
            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            d_real = D(real)
            loss_d_real = criterion(d_real, real_labels)

            z = torch.randn(bsz, Z_DIM, 1, 1, device=device)
            fake = G(z)
            d_fake = D(fake.detach())
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            G.zero_grad()
            d_fake2 = D(fake)
            loss_g = criterion(d_fake2, real_labels)
            loss_g.backward()
            opt_g.step()

            ep_g += loss_g.item()
            ep_d += loss_d.item()

            if i % 100 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] ep{epoch} it{i}/{len(loader)} lossD={loss_d.item():.4f} lossG={loss_g.item():.4f}", flush=True)

        avg_g = ep_g / len(loader)
        avg_d = ep_d / len(loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] EPOCH {epoch}/{EPOCHS} avg_lossG={avg_g:.4f} avg_lossD={avg_d:.4f} time={dt:.0f}s", flush=True)

        G.eval()
        with torch.no_grad():
            fake_imgs = G(fixed_z).cpu()
        G.train()
        fake_imgs = (fake_imgs + 1) / 2
        fake_imgs = torch.clamp(fake_imgs, 0, 1)

        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for ax, img in zip(axes.flatten(), fake_imgs):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis("off")
        plt.suptitle(f"v2 Epoch {epoch} — LR={LR} Subset={SUBSET}")
        plt.tight_layout()
        plt.savefig(f"samples_v2/samples_ep{epoch:02d}.png", dpi=80, bbox_inches="tight")
        plt.close()

        torch.save(G.state_dict(), "checkpoints/generator_v2.pt")
        torch.save(D.state_dict(), "checkpoints/discriminator_v2.pt")

    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"v2 Training Losses (LR={LR}, {SUBSET} imgs, {EPOCHS} epochs)")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve_v2.png", dpi=100, bbox_inches="tight")
    plt.close()

    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
