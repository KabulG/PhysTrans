import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_series(base_dir):
    pred = np.load(os.path.join(base_dir, "pred.npy"))
    true = np.load(os.path.join(base_dir, "true.npy"))
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    true = true.reshape(-1, true.shape[-2], true.shape[-1])
    pred = pred[:, :, -1].reshape(-1)
    true = true[:, :, -1].reshape(-1)
    return pred, true

def plot_scatter(trues, preds, out_path, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, alpha=0.1, s=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="r", linewidth=2)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hist(residual, out_path, title):
    plt.figure(figsize=(8, 4))
    plt.hist(residual, bins=100, density=True)
    plt.title(title)
    plt.xlabel("pred - true")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_bad_windows(trues_flat, preds_flat, pred_len, out_dir, title_prefix, topk=3):
    n = trues_flat.shape[0] // pred_len
    tr = trues_flat[: n * pred_len].reshape(n, pred_len)
    pr = preds_flat[: n * pred_len].reshape(n, pred_len)
    mse = np.mean((pr - tr) ** 2, axis=1)
    idx = np.argsort(mse)[-topk:][::-1]
    for rank, i in enumerate(idx):
        plt.figure(figsize=(10, 3))
        plt.plot(tr[i], label="true")
        plt.plot(pr[i], label="pred")
        plt.legend()
        plt.title(f"{title_prefix} worst window {rank+1} (idx={i}, mse={mse[i]:.4f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{title_prefix}_worst_{rank+1}.png"))
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--shortcut_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pred_len", type=int, default=96)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    b_pred, b_true = load_series(args.baseline_dir)
    s_pred, s_true = load_series(args.shortcut_dir)
    plot_scatter(b_true, b_pred, os.path.join(args.out_dir, "baseline_scatter.png"), "baseline scatter")
    plot_scatter(s_true, s_pred, os.path.join(args.out_dir, "shortcut_scatter.png"), "shortcut scatter")
    plot_hist(b_pred - b_true, os.path.join(args.out_dir, "baseline_residual_hist.png"), "baseline residual")
    plot_hist(s_pred - s_true, os.path.join(args.out_dir, "shortcut_residual_hist.png"), "shortcut residual")
    plot_bad_windows(b_true, b_pred, args.pred_len, args.out_dir, "baseline", topk=3)
    plot_bad_windows(s_true, s_pred, args.pred_len, args.out_dir, "shortcut", topk=3)

if __name__ == "__main__":
    main()

