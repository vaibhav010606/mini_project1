"""Train the video autoencoders on NORMAL UCSD Ped2 training clips only.

    python video_model/train.py --model big
    python video_model/train.py --model student --distill   # edge model, learns from big model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG_DIR, MODEL_DIR, plot_style, set_seed  # noqa: E402
from video_model.dataset import load_ped2, make_clips  # noqa: E402
from video_model.models import MODELS, n_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="big")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--distill", action="store_true", help="student also imitates the big model")
    args = ap.parse_args()

    set_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, _, _ = load_ped2()
    clips = np.concatenate([make_clips(v) for v in train])
    X = torch.tensor(clips, device=dev)
    print(f"[{args.model}] training clips={len(X)} shape={tuple(X.shape[1:])} device={dev}")

    model = MODELS[args.model]().to(dev)
    teacher = None
    if args.distill:
        teacher = MODELS["big"]().to(dev)
        teacher.load_state_dict(torch.load(MODEL_DIR / "video_big.pt", map_location=dev))
        teacher.eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    losses, t0 = [], time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            x = X[perm[i:i + args.batch]]
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[3])
            out = model(x)
            loss = F.mse_loss(out, x)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
                loss = 0.5 * loss + 0.5 * F.mse_loss(out, t_out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        losses.append(tot / len(X))
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep + 1:3d}/{args.epochs}  loss={losses[-1]:.5f}  ({time.time() - t0:.0f}s)")

    torch.save(model.state_dict(), MODEL_DIR / f"video_{args.model}.pt")
    info = dict(model=args.model, params=n_params(model), epochs=args.epochs, final_loss=losses[-1],
                train_seconds=time.time() - t0, distill=args.distill)
    (MODEL_DIR / f"video_{args.model}_train.json").write_text(json.dumps(info, indent=2))

    plt = plot_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction loss (MSE)")
    ax.set_title(f"Training loss: {args.model} autoencoder")
    fig.savefig(FIG_DIR / f"video_loss_{args.model}.png")
    plt.close(fig)
    print(f"Saved model ({n_params(model):,} params) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
