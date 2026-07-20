"""
Recompute every reported quantity from saved predictions — no retraining.

`train.py` stores the ground truth and the positive-class probability of each run, so all of the
following can be derived afterwards, and re-derived by anyone checking the work:

  * the seven metrics, mean +/- SD over seeds
  * the prevalence floor, against which AUC-PR must be read
  * the operating point: specificity at a fixed sensitivity, rather than at the default 0.5
    threshold, which is the comparison a screening deployment actually cares about
  * optionally, a significance comparison against one reference model

Usage:
  python evaluate.py --runs runs
  python evaluate.py --runs runs --sensitivity 0.90
  python evaluate.py --runs runs --compare_to DiffSA
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, auc as sk_auc)

MET = ["accuracy", "balanced_accuracy", "recall", "specificity", "f1", "auc_pr", "auc"]
HDR = {"accuracy": "Acc", "balanced_accuracy": "Bal-Acc", "recall": "Recall", "specificity": "Spec",
       "f1": "F1", "auc_pr": "AUC-PR", "auc": "AUROC"}


def metrics_from_probs(y, p1):
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


def specificity_at_sensitivity(y, p1, target):
    """Move the decision threshold until sensitivity reaches `target`, then read specificity.

    This is one horizontal slice of the ROC curve: find where the curve crosses TPR = target and
    report 1 - FPR there. The default 0.5 threshold has no clinical meaning on an imbalanced
    corpus, whereas a required sensitivity is set by the screening question.
    """
    fpr, tpr, _ = roc_curve(y, p1)
    idx = np.searchsorted(tpr, target, side="left")
    if idx >= len(tpr):
        return float("nan")
    if idx == 0 or tpr[idx] == target:
        return float(1.0 - fpr[idx])
    # linear interpolation between the two bracketing ROC points
    w = (target - tpr[idx - 1]) / (tpr[idx] - tpr[idx - 1])
    return float(1.0 - (fpr[idx - 1] + w * (fpr[idx] - fpr[idx - 1])))


def bh(p):
    """Benjamini-Hochberg step-up FDR correction."""
    p = np.asarray(p, float)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for r, i in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[i] * n / (n - r))
        q[i] = prev
    return q


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def load_runs(runs_dir):
    """{model: {"per_run": {metric: [...]}, "pooled": (y, p1), "n": int}}"""
    out = {}
    for model_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        files = sorted(glob.glob(os.path.join(model_dir, "*_probs.npz")))
        if not files:
            continue
        per = {m: [] for m in MET}
        ys, ps = [], []
        for f in files:
            z = np.load(f)
            v = metrics_from_probs(z["y"], z["p1"])
            for m in MET:
                per[m].append(v[m])
            ys.append(z["y"])
            ps.append(z["p1"])
        out[os.path.basename(model_dir)] = {
            "per_run": per, "n": len(files),
            "pooled": (np.concatenate(ys), np.concatenate(ps)),
            "prevalence": float(np.mean(np.concatenate(ys))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="directory written by train.py")
    ap.add_argument("--sensitivity", type=float, default=0.90,
                    help="fixed sensitivity for the operating-point table")
    ap.add_argument("--compare_to", default=None,
                    help="model name to test every other model against (Mann-Whitney U + BH-FDR)")
    ap.add_argument("--json_out", default=None, help="also write the results as JSON")
    args = ap.parse_args()

    R = load_runs(args.runs)
    if not R:
        raise SystemExit(f"no runs found under {args.runs}/ — run train.py first")
    models = list(R)

    # ---- metrics ----
    print(f"\n## Metrics (mean +/- SD over seeds, threshold 0.5)\n")
    print("| Model | n | " + " | ".join(HDR[m] for m in MET) + " |")
    print("|---" * (len(MET) + 2) + "|")
    best = {m: max(models, key=lambda k: np.mean(R[k]["per_run"][m])) for m in MET}
    for k in models:
        cells = []
        for m in MET:
            a = np.array(R[k]["per_run"][m]) * 100
            s = f"{a.mean():.1f}+/-{a.std(ddof=1):.1f}" if len(a) > 1 else f"{a.mean():.1f}"
            cells.append(f"**{s}**" if k == best[m] else s)
        print(f"| {k} | {R[k]['n']} | " + " | ".join(cells) + " |")

    prev = np.mean([R[k]["prevalence"] for k in models])
    print(f"\n> Positive-class prevalence of the test set: **{prev:.3f}**. This is the AUC-PR floor —"
          f" the score a no-skill classifier reaches — so AUC-PR must be read relative to it and is"
          f" not comparable across corpora with different prevalence.")

    # ---- operating point ----
    print(f"\n## Operating point: specificity at sensitivity = {args.sensitivity:.0%}\n")
    print("| Model | Specificity | Recall @0.5 |")
    print("|---|---|---|")
    op = {}
    for k in models:
        y, p1 = R[k]["pooled"]
        op[k] = specificity_at_sensitivity(y, p1, args.sensitivity)
        r05 = np.mean(R[k]["per_run"]["recall"]) * 100
        print(f"| {k} | {op[k]*100:.1f} | {r05:.1f} |")
    print(f"\n> Ranking models at the default 0.5 threshold and at a fixed clinical operating point"
          f" are different comparisons, and they need not agree.")

    # ---- significance ----
    sig = {}
    if args.compare_to:
        ref = args.compare_to
        if ref not in R:
            raise SystemExit(f"--compare_to {ref} not found; available: {', '.join(models)}")
        others = [k for k in models if k != ref]
        print(f"\n## Significance vs {ref} (Mann-Whitney U, Benjamini-Hochberg FDR within each metric)\n")
        print("| Model | " + " | ".join(HDR[m] for m in MET) + " |")
        print("|---" * (len(MET) + 1) + "|")
        qs = {}
        for m in MET:
            a = np.array(R[ref]["per_run"][m])
            ps = [stats.mannwhitneyu(a, np.array(R[k]["per_run"][m]),
                                     alternative="two-sided").pvalue for k in others]
            qs[m] = dict(zip(others, bh(ps)))
        for k in others:
            row = []
            for m in MET:
                q = qs[m][k]
                d = cohen_d(np.array(R[k]["per_run"][m]), np.array(R[ref]["per_run"][m]))
                row.append(f"{'**' if q < .05 else ''}q={q:.3f}{'**' if q < .05 else ''} (d={d:+.2f})")
            sig[k] = {m: {"q": float(qs[m][k]),
                          "d": cohen_d(np.array(R[k]["per_run"][m]),
                                       np.array(R[ref]["per_run"][m]))} for m in MET}
            print(f"| {k} | " + " | ".join(row) + " |")
        print(f"\n> A metric with q >= .05 is a tie, even if one mean is nominally higher; report it"
              f" as comparable rather than as a win. `d` is signed relative to {ref}.")

    if args.json_out:
        json.dump({"per_run": {k: R[k]["per_run"] for k in models},
                   "n_runs": {k: R[k]["n"] for k in models},
                   "prevalence": prev,
                   f"specificity_at_sens_{args.sensitivity}": op,
                   "significance": sig},
                  open(args.json_out, "w"), indent=1)
        print(f"\n[written] {args.json_out}")


if __name__ == "__main__":
    main()
