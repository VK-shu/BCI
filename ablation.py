"""Ablation study for Table 3 of the report.

Runs the same LOSO evaluation as main.py under several processing configurations and
prints / saves the mean accuracy of every model (the best one is what Table 3 needs).

Usage:  uv run python ablation.py
"""
import numpy as np
import main
from main import Config

# (name, USE_FILTERING, USE_FREQ_FEATURES, USE_TIME_FEATURES)
CONFIGS = [
    ("Raw time-domain samples",        False, False, False),
    ("+ Filtering (raw samples)",      True,  False, False),
    ("+ Frequency features",           True,  True,  False),
    ("+ Time features (final)",        True,  True,  True),
]
MODELS = ['MLP', 'SVM', 'RF', 'SelfSelected']


def run_config(use_filter, use_freq, use_time):
    Config.USE_FILTERING, Config.USE_FREQ_FEATURES, Config.USE_TIME_FEATURES = use_filter, use_freq, use_time
    X, y, subjects = main.load_all_subjects()
    if X is None:
        raise SystemExit("Dataset not found / invalid. Run check_data.py first.")
    res = {}
    for m in MODELS:
        r = main.leave_one_subject_out_validation(m, X, y, subjects)
        res[m] = (np.mean(r['accuracies']), np.std(r['accuracies']))
    return X.shape[1], res


def main_ablation():
    lines = []
    header = f"{'Configuration':<30}{'#feat':>7}" + "".join(f"{m:>16}" for m in MODELS) + f"{'Best':>16}"
    lines.append(header)
    for name, f, fr, t in CONFIGS:
        n_feat, res = run_config(f, fr, t)
        best = max(res, key=lambda m: res[m][0])
        row = f"{name:<30}{n_feat:>7}" + "".join(f"{res[m][0]:>9.3f}+/-{res[m][1]:.3f}" for m in MODELS)
        row += f"{best + ' ' + format(res[best][0], '.3f'):>16}"
        lines.append(row)
        print(row, flush=True)
    text = "\n".join(lines)
    print("\n" + text)
    with open("ablation_results.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\nSaved to ablation_results.txt")


if __name__ == "__main__":
    main_ablation()
