"""Ablation study for Table 4 / Appendix B of the report (separate from main.py).

Each row runs the same LOSO evaluation as main.py with a different set of processing
switches. The rows are strictly cumulative: every row adds exactly one step to the row
above, and the subject-wise z-score stays OFF until the last step, so the reference
row really is "raw samples, no filtering, no normalization".

    row 1  raw 8 s samples                      (no filter, no features, no z-score)
    row 2  + band-pass / notch filtering
    row 3  + window band-power features
    row 4  + window time-domain features (log std, kurtosis)
    row 5  + subject-wise z-score               (= final pipeline of main.py)
    extra  band-power features + z-score, without time-domain features

Usage:  uv run python ablation.py        (writes ablation_results.txt)
"""
import contextlib
import io
import warnings

import numpy as np

import main
from main import Config

warnings.filterwarnings("ignore")

# (name, USE_FILTERING, USE_FREQ_FEATURES, USE_TIME_FEATURES, SUBJECT_ZSCORE)
CONFIGS = [
    ("1 Raw samples",                     False, False, False, False),
    ("2 + Filtering (raw samples)",       True,  False, False, False),
    ("3 + Window band-power features",    True,  True,  False, False),
    ("4 + Time-domain features",          True,  True,  True,  False),
    ("5 + Subject z-score (final)",       True,  True,  True,  True),
    ("(extra) band-power + z-score",      True,  True,  False, True),
]
MODELS = ["MLP", "SVM", "RF", "SelfSelected"]


def run_config(use_filter, use_freq, use_time, use_z):
    Config.USE_FILTERING = use_filter
    Config.USE_FREQ_FEATURES = use_freq
    Config.USE_TIME_FEATURES = use_time
    Config.SUBJECT_ZSCORE = use_z
    with contextlib.redirect_stdout(io.StringIO()):          # silence the loader
        X, y, subjects = main.load_all_subjects()
    if X is None:
        raise SystemExit("Dataset not found / invalid.")
    res = {}
    for m in MODELS:
        acc = main.leave_one_subject_out_validation(m, X, y, subjects)["accuracies"]
        res[m] = (np.mean(acc), np.std(acc))
    return X.shape[1], res


def main_ablation():
    saved = (Config.USE_FILTERING, Config.USE_FREQ_FEATURES, Config.USE_TIME_FEATURES, Config.SUBJECT_ZSCORE)
    lines = [f"{'Configuration':<34}{'#feat':>6}" + "".join(f"{m:>18}" for m in MODELS) + f"{'Best':>18}"]
    try:
        for name, f, fr, t, z in CONFIGS:
            n_feat, res = run_config(f, fr, t, z)
            best = max(res, key=lambda m: res[m][0])
            row = f"{name:<34}{n_feat:>6}" + "".join(f"{res[m][0]:>10.3f}+/-{res[m][1]:.3f}" for m in MODELS)
            row += f"{best + ' ' + format(res[best][0], '.3f'):>18}"
            lines.append(row)
            print(row, flush=True)
    finally:
        Config.USE_FILTERING, Config.USE_FREQ_FEATURES, Config.USE_TIME_FEATURES, Config.SUBJECT_ZSCORE = saved
    with open("ablation_results.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nSaved to ablation_results.txt")


if __name__ == "__main__":
    main_ablation()
