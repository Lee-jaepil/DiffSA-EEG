"""
Generate a small synthetic dataset in the exact format train.py expects.

This exists so the repository can be run end to end without any data access: the shapes, the code
path, and the output files are identical to a real experiment. The signals themselves are synthetic
and carry no clinical meaning — 1/f background noise, with a slow rhythm added for the positive
class. The effect is much cleaner than any real clinical one, deliberately, so that a short demo
run separates the classes.

The clinical corpora used in the paper (TUAB, TUEP) are distributed by the Neural Engineering Data
Consortium under an agreement that does not permit redistribution, so no real recording is included
here. See README.md for how to obtain them.

Usage:
  python sample_data/make_sample_data.py                        # defaults below
  python sample_data/make_sample_data.py --n_train 80 --amplitude 0.6   # bigger and harder

out: sample_data/train/sample.npz  and  sample_data/test.npz  (~150 MB at the defaults)
"""
import argparse
import os

import numpy as np

N_WINDOWS = 120          # windows per recording
N_CHANNELS = 21
N_SAMPLES = 250          # 1 s at 250 Hz
FS = 250.0


def make_recordings(n, positive_rate, amplitude, rng):
    """Return (data, labels) with data shaped (n, 120, 21, 250) float32.

    The positive count is fixed rather than drawn, so a small sample still has both classes
    present in a usable proportion.
    """
    n_pos = int(np.clip(round(n * positive_rate), 2, n - 2))
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, size=n_pos, replace=False)] = 1
    X = np.empty((n, N_WINDOWS, N_CHANNELS, N_SAMPLES), dtype=np.float32)
    t = np.arange(N_SAMPLES) / FS

    for i in range(n):
        # 1/f-shaped background, the dominant feature of resting EEG
        white = rng.standard_normal((N_WINDOWS, N_CHANNELS, N_SAMPLES))
        spec = np.fft.rfft(white, axis=-1)
        freqs = np.fft.rfftfreq(N_SAMPLES, 1 / FS)
        spec /= np.maximum(freqs, 1.0)
        x = np.fft.irfft(spec, n=N_SAMPLES, axis=-1)

        if y[i] == 1:
            # Positive class: a slow rhythm added over a subset of channels. The amplitude is set
            # so that a short demo run separates the classes; it is far cleaner than any real
            # clinical effect and is not meant to be realistic.
            chans = rng.choice(N_CHANNELS, size=8, replace=False)
            f0 = rng.uniform(3.0, 7.0)
            burst = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)) * amplitude
            x[:, chans] += burst

        # per-recording z-score, as in the real preprocessing
        x = (x - x.mean()) / (x.std() + 1e-8)
        X[i] = x.astype(np.float32)

    return X, y


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    # One recording is ~2.5 MB, so these defaults produce ~150 MB on disk. They are the smallest
    # sizes at which the minority class is large enough for the metrics to mean anything.
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=24)
    ap.add_argument("--positive_rate", type=float, default=0.21,
                    help="fraction of positive recordings; the default mirrors the epilepsy "
                         "corpus's natural prevalence, so the class imbalance is realistic")
    ap.add_argument("--amplitude", type=float, default=1.2,
                    help="strength of the positive-class rhythm; lower makes the task harder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=here)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)

    Xtr, ytr = make_recordings(args.n_train, args.positive_rate, args.amplitude, rng)
    np.savez_compressed(os.path.join(args.out, "train", "sample.npz"), data=Xtr, labels=ytr)

    Xte, yte = make_recordings(args.n_test, args.positive_rate, args.amplitude, rng)
    np.savez_compressed(os.path.join(args.out, "test.npz"), data=Xte, labels=yte)

    print(f"train: {Xtr.shape} {np.bincount(ytr, minlength=2).tolist()} -> {args.out}/train/sample.npz")
    print(f"test : {Xte.shape} {np.bincount(yte, minlength=2).tolist()} -> {args.out}/test.npz")
    print("\nnow run, for example:")
    print("  python train.py --model DiffSA --train_dir sample_data/train "
          "--test_npz sample_data/test.npz --epochs 10 --batch_size 4")
    print("\nNote: with a sample this small the seed-to-seed spread is large. Use several seeds and"
          " read the mean, exactly as the paper does.")


if __name__ == "__main__":
    main()
