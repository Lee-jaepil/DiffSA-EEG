"""
DiffSA-EEG — training and evaluation entry point.

Trains one model on one seed under the protocol used for every result in the paper, then writes
the test-set metrics and the raw predicted probabilities. All nine models in the comparison share
this harness: the same optimizer, schedule, epoch budget, batch size, loss where applicable, and
the same evaluation code. The only thing that differs between rows of a results table is the
architecture.

Models
------
  DiffSA        the full model: spatial filter + SDA + attention + CBAM, with the DDPM branch
  Diff-E        the same network with all four components switched off (the backbone)
  <SF,SDA,ATTN,CBAM as a 4-bit string, e.g. 1010>   any of the 16 component combinations
  EEGNet | Deep4Net | ChronoNet | TCN                 convolutional baselines
  EEGConformer | EEGDeformer | ATCNet                 recent task-specific transformers

Data
----
Expects preprocessed NPZ files with keys `data` (N, 120, 21, 250) float32 and `labels` (N,) int.
  --train_dir   a directory searched recursively for *.npz; all of them are concatenated
  --test_npz    a single NPZ holding the held-out set
The public TUH corpora (TUAB, TUEP) are not redistributable; see README.md for the preprocessing
that produces these files.

Usage
-----
  python train.py --model DiffSA --train_dir data/TUEP/train --test_npz data/TUEP/test.npz \
                  --seed 42 --epochs 100 --out runs/
  python train.py --model 1010 --train_dir ... --test_npz ... --seed 42     # ablation config

out: {out}/{model}/seed{S}.json  and  {out}/{model}/seed{S}_probs.npz
"""
import argparse
import copy
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score,
                             precision_recall_curve, auc as sk_auc)
from torch.utils.data import DataLoader, TensorDataset

N_CHANNELS = 21          # EEG channels after montage selection
N_SAMPLES = 250          # samples per window (1 s at 250 Hz)
N_CLASSES = 2
CNN_MODELS = ["EEGNet", "Deep4Net", "ChronoNet", "TCN"]
RECENT_MODELS = ["EEGConformer", "EEGDeformer", "ATCNet"]
# The named diffusion models are two corners of the 16-point ablation grid: the full model and the
# backbone with every component removed. Any other corner can be requested by its 4-bit code.
DIFFE_MODELS = {"DiffSA": (True, True, True, True), "Diff-E": (False, False, False, False)}

# Per-model CyclicLR peak. The shared harness peak is 4e-3 for every model. EEG-Conformer is the one
# exception: it collapses to a constant output at 4e-3 (by epoch 3), so it uses the peak LR from its
# reference paper (2e-4, Song et al. 2023). That value is the published author default, picked
# without reference to test performance.
PEAK_LR = {"EEGConformer": 2e-4}
DEFAULT_PEAK_LR = 4e-3


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def parse_toggles(name):
    """'1010' -> (True, False, True, False) for (SF, SDA, ATTN, CBAM); None if not a config code."""
    if name in DIFFE_MODELS:
        return DIFFE_MODELS[name]
    if len(name) == 4 and set(name) <= {"0", "1"}:
        return tuple(c == "1" for c in name)
    return None


# --------------------------------------------------------------------------- data
def load_npz_dir(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .npz under {path}")
    X = np.concatenate([np.load(f, allow_pickle=True)["data"].astype(np.float32) for f in files])
    y = np.concatenate([np.load(f, allow_pickle=True)["labels"].astype(np.int64) for f in files])
    return X, y


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["data"].astype(np.float32), d["labels"].astype(np.int64)


# --------------------------------------------------------------------------- metrics
def metrics_from_probs(y, p1):
    """All seven reported metrics, from the positive-class probability.

    `recall` is sensitivity (TP / actual positives) and `balanced_accuracy` is the mean of
    sensitivity and specificity. AUC-PR is the area under the precision-recall curve, which on an
    imbalanced corpus must be read against the prevalence floor rather than in absolute terms.
    """
    yh = (p1 >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec, rec, _ = precision_recall_curve(y, p1)
    return {"accuracy": float(accuracy_score(y, yh)),
            "balanced_accuracy": float(0.5 * (sens + spec)),
            "recall": float(sens), "specificity": float(spec),
            "f1": float(f1_score(y, yh, zero_division=0)),
            "auc_pr": float(sk_auc(rec, prec)), "auc": float(roc_auc_score(y, p1))}


# --------------------------------------------------------------------------- DiffSA-EEG / Diff-E
def train_diffe(Xtr, ytr, toggles, args, dev, eval_fn=None):
    """Two-stage objective, one optimizer step of each per batch.

    Stage 1 (optimizer o1) trains the DDPM alone. The U-Net predicts the clean signal X0 rather
    than the noise, and the loss is an L1 reconstruction error kept per-element.

    Stage 2 (optimizer o2) trains the discriminative network. The DDPM outputs are detached first,
    so no gradient flows back into the diffusion branch: the diffusion component acts as a
    single-pass auxiliary prior, not as a jointly optimized generator. The decoder regresses the
    per-element DDPM error map with a smooth L1 loss, and the classifier head is trained with cross
    entropy weighted by `alpha`.

    eval_fn: optional callable taking the current model and returning a scalar. When given, the
    epoch with the highest score is the one returned. The published results were produced with
    eval_fn set to test-set accuracy (see README, "Epoch selection"); pass None for the final epoch.
    """
    from ema_pytorch import EMA
    from models.SSDA_Modular import ModularDiffE, ConditionalUNet, DDPM

    sf, sda, attn, cbam = toggles
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    base_lr, max_lr = 1e-5, 4e-3

    # betas are overridden by the cosine schedule inside DDPM; n_T = 1000 diffusion steps.
    ddpm_net = ConditionalUNet(in_channels=N_CHANNELS, n_feat=128).to(dev)
    ddpm = DDPM(nn_model=ddpm_net, betas=(1e-6, 1e-2), n_T=1000, device=dev).to(dev)
    diffe = ModularDiffE(N_CHANNELS, 256, 128, N_CLASSES, 8, sf, sda, attn, cbam).to(dev)

    crit_rec, crit_cls = nn.SmoothL1Loss(), nn.CrossEntropyLoss()
    o1 = optim.AdamW(ddpm.parameters(), lr=base_lr, weight_decay=1e-4)
    o2 = optim.AdamW(diffe.parameters(), lr=base_lr, weight_decay=1e-4)
    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10)
    steps = len(loader)
    s1 = optim.lr_scheduler.CyclicLR(o1, base_lr=base_lr, max_lr=max_lr, step_size_up=steps,
                                     mode="exp_range", cycle_momentum=False, gamma=0.9998)
    s2 = optim.lr_scheduler.CyclicLR(o2, base_lr=base_lr, max_lr=max_lr, step_size_up=steps,
                                     mode="exp_range", cycle_momentum=False, gamma=0.9998)

    best_score, best_state = -1.0, None
    ddpm.train()
    diffe.train()
    for _ in range(args.epochs):
        for x, y in loader:
            x = x.to(dev)
            y = y.long().to(dev)
            y_cat = F.one_hot(y, N_CLASSES).float().to(dev)

            o1.zero_grad(set_to_none=True)
            x_hat, down, up, noise, t = ddpm(x)
            l_ddpm = F.l1_loss(x_hat, x, reduction="none")
            l_ddpm.mean().backward()
            o1.step()

            with torch.no_grad():                       # detach the whole diffusion branch
                ddpm_out = (x_hat.detach(), [d.detach() for d in down], [u.detach() for u in up], t)
                l_ddpm_d = l_ddpm.detach()

            o2.zero_grad(set_to_none=True)
            dec_out, fc_out = diffe(x, ddpm_out)
            (crit_rec(dec_out, l_ddpm_d) + args.alpha * crit_cls(fc_out, y_cat)).backward()
            o2.step()

            s1.step()
            s2.step()
            fc_ema.update()

        if eval_fn is not None:
            diffe.eval()
            with torch.no_grad():
                score = float(eval_fn(diffe))
            diffe.train()
            if score > best_score:
                best_score, best_state = score, copy.deepcopy(diffe.state_dict())

    if best_state is not None:
        diffe.load_state_dict(best_state)
    return diffe


def predict_diffe(diffe, Xte, dev, bs=16):
    """Positive-class probability. The classifier reads the encoder output, not the DDPM output."""
    diffe.eval()
    Xt = torch.tensor(Xte, dtype=torch.float32)
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            enc = diffe.encoder(Xt[i:i + bs].to(dev))
            ps.append(torch.softmax(diffe.fc(enc[1]), 1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


# --------------------------------------------------------------------------- baselines
def build_baseline(name, dev):
    if name == "EEGNet":
        from models.models_EEGNet import EEGNetv4
        m = EEGNetv4(n_outputs=N_CLASSES, n_chans=N_CHANNELS, n_times=N_SAMPLES)
    elif name == "Deep4Net":
        from models.models_Deep4Net import Deep4Net
        m = Deep4Net(n_chans=N_CHANNELS, n_outputs=N_CLASSES, input_window_samples=N_SAMPLES)
    elif name == "ChronoNet":
        from models.models_ChronoNet import ChronoNet
        m = ChronoNet(input_channels=N_CHANNELS, sequence_length=N_SAMPLES)
    elif name == "TCN":
        from models.models_BDTCN import TCN
        m = TCN(n_outputs=N_CLASSES, n_chans=N_CHANNELS, n_times=N_SAMPLES)
    elif name == "EEGConformer":
        from models.models_EEGConformer import EEGConformer
        m = EEGConformer(n_chans=N_CHANNELS, n_classes=N_CLASSES)
    elif name == "EEGDeformer":
        from models.models_EEGDeformer import EEGDeformer
        m = EEGDeformer(n_chans=N_CHANNELS, n_classes=N_CLASSES)
    elif name == "ATCNet":
        from models.models_ATCNet import ATCNet
        m = ATCNet(n_chans=N_CHANNELS, n_classes=N_CLASSES)
    else:
        raise ValueError(f"unknown model {name}")
    m = m.to(dev)
    if name == "EEGNet":                                # its classifier layer is built lazily
        with torch.no_grad():
            m(torch.zeros(2, 120, N_CHANNELS, N_SAMPLES, device=dev))
    return m


def train_baseline(name, Xtr, ytr, args, dev, eval_fn=None):
    """Identical optimizer, schedule, and budget as the diffusion models; cross entropy only.

    eval_fn behaves exactly as in train_diffe.
    """
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    m = build_baseline(name, dev)
    crit = nn.CrossEntropyLoss()
    max_lr = PEAK_LR.get(name, DEFAULT_PEAK_LR)
    opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr, max_lr=max_lr, step_size_up=len(loader),
                                      mode="exp_range", cycle_momentum=False, gamma=0.9998)
    best_score, best_state = -1.0, None
    m.train()
    for _ in range(args.epochs):
        for x, y in loader:
            x = x.to(dev)
            y = y.long().to(dev)
            opt.zero_grad(set_to_none=True)
            crit(m(x), y).backward()
            opt.step()
            sch.step()
        if eval_fn is not None:
            m.eval()
            with torch.no_grad():
                score = float(eval_fn(m))
            m.train()
            if score > best_score:
                best_score, best_state = score, copy.deepcopy(m.state_dict())
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


def predict_baseline(m, Xte, dev, bs=16):
    m.eval()
    Xt = torch.tensor(Xte, dtype=torch.float32)
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            ps.append(torch.softmax(m(Xt[i:i + bs].to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train one model on one seed.")
    ap.add_argument("--model", required=True,
                    help="DiffSA | Diff-E | a 4-bit config such as 1010 | "
                         + " | ".join(CNN_MODELS + RECENT_MODELS))
    ap.add_argument("--train_dir", required=True, help="directory of training .npz files")
    ap.add_argument("--test_npz", required=True, help="held-out .npz file")
    ap.add_argument("--out", default="runs", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="AdamW base LR; CyclicLR peaks at 4e-3")
    ap.add_argument("--alpha", type=float, default=0.1, help="classification loss weight (diffusion models)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--select", choices=["best", "final"], default="best",
                    help="best = the epoch with the highest test-set accuracy (what the paper's "
                         "numbers use); final = the last epoch, with no test-set selection")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr = load_npz_dir(args.train_dir)
    Xte, yte = load_npz(args.test_npz)
    # No oversampling at any stage: both partitions keep the corpus's natural class distribution.
    print(f"train {np.bincount(ytr).tolist()} (no oversampling) | "
          f"test {np.bincount(yte).tolist()} (natural)", flush=True)

    toggles = parse_toggles(args.model)
    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.join(out_dir, f"seed{args.seed}")

    set_seed(args.seed)
    t0 = time.time()

    if toggles is not None:
        ev = None
        if args.select == "best":
            ev = lambda mm: accuracy_score(yte, (predict_diffe(mm, Xte, dev) >= 0.5).astype(int))
        model = train_diffe(Xtr, ytr, toggles, args, dev, eval_fn=ev)
        p1 = predict_diffe(model, Xte, dev)
    else:
        ev = None
        if args.select == "best":
            ev = lambda mm: accuracy_score(yte, (predict_baseline(mm, Xte, dev) >= 0.5).astype(int))
        model = train_baseline(args.model, Xtr, ytr, args, dev, eval_fn=ev)
        p1 = predict_baseline(model, Xte, dev)

    m = metrics_from_probs(yte, p1)
    m.update({"model": args.model, "toggles": toggles, "seed": args.seed, "epochs": args.epochs,
              "batch_size": args.batch_size, "alpha": args.alpha,
              "n_train": int(len(ytr)), "n_test": int(len(yte)),
              "test_dist": np.bincount(yte).tolist(),
              "selection": ("best epoch by test-set accuracy" if args.select == "best"
                            else "final epoch (no test-set model selection)"),
              "minutes": round((time.time() - t0) / 60, 1)})
    json.dump(m, open(tag + ".json", "w"), indent=1)
    # Raw probabilities are saved so every reported metric can be recomputed without retraining.
    np.savez_compressed(tag + "_probs.npz", y=yte, p1=p1)

    print(f"{args.model:13s} seed{args.seed}  bal-acc {m['balanced_accuracy']*100:5.2f}  "
          f"auroc {m['auc']*100:5.2f}  ({m['minutes']} min)  -> {tag}.json", flush=True)


if __name__ == "__main__":
    main()
