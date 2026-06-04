import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_series_any(base_dir):
    cand = [os.path.join(base_dir, "pred.npy")]
    if "results" in base_dir:
        alt_dir = base_dir.replace("\\results\\", "\\test_results\\").replace("/results/", "/test_results/")
        cand.append(os.path.join(alt_dir, "pred.npy"))
    elif "test_results" in base_dir:
        alt_dir = base_dir.replace("\\test_results\\", "\\results\\").replace("/test_results/", "/results/")
        cand.append(os.path.join(alt_dir, "pred.npy"))

    pred_path = next((p for p in cand if os.path.exists(p)), None)
    if pred_path is None:
        raise FileNotFoundError(f"pred.npy not found under {base_dir}")

    true_path = os.path.join(os.path.dirname(pred_path), "true.npy")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"true.npy not found under {os.path.dirname(pred_path)}")

    pred = np.load(pred_path)
    true = np.load(true_path)
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    true = true.reshape(-1, true.shape[-2], true.shape[-1])
    pred = pred[:, :, -1].reshape(-1)
    true = true[:, :, -1].reshape(-1)
    return pred, true

def plot_series(pred, true, out_path, title):
    plt.figure(figsize=(18, 4))
    plt.plot(true, label="true", linewidth=1.0)
    plt.plot(pred, label="pred", linewidth=1.0)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def resolve_base_dir(test_dir):
    test_dir = test_dir.rstrip(os.sep)
    base = os.path.basename(test_dir)
    root = os.path.dirname(os.path.dirname(test_dir))
    rd = os.path.join(root, "results", base)
    td = os.path.join(root, "test_results", base)
    if os.path.exists(os.path.join(rd, "pred.npy")):
        return rd
    if os.path.exists(os.path.join(td, "pred.npy")):
        return td
    return rd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dirs", nargs="+", required=True)
    parser.add_argument("--out_files", nargs="+", required=True)
    args = parser.parse_args()
    assert len(args.test_dirs) == len(args.out_files)
    for td, of in zip(args.test_dirs, args.out_files):
        rd = resolve_base_dir(td)
        pred, true = load_series_any(rd)
        title = os.path.basename(rd)
        os.makedirs(os.path.dirname(of), exist_ok=True)
        plot_series(pred, true, of, title)

if __name__ == "__main__":
    main()
