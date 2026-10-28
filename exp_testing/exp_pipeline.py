"""Pipeline experiments reported in Appendix B: segment length / overlap sweep and feature selection.

These two sweeps change the *pipeline* (Config.SEGMENT_LENGTH, OVERLAP_RATIO, FEATURE_SELECTION) rather than a
classifier setting, so they run through main.py's own LOSO function. main.py is imported only; it is not modified.
Usage (from the project root):   uv run python exp_testing/exp_pipeline.py
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import main                      # noqa: E402
from main import Config          # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["MLP", "SVM", "RF", "SelfSelected"]


def evaluate():
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, s = main.load_all_subjects()
    acc = [np.mean(main.leave_one_subject_out_validation(m, X, y, s)["accuracies"]) for m in MODELS]
    return len(y) // 3, acc


def main_exp():
    import sklearn
    lines = ["Pipeline experiments (segment length / overlap, feature selection)", "=" * 66,
             f"Reproducibility: random_state = {Config.RANDOM_STATE}; python {sys.version.split()[0]}, "
             f"numpy {np.__version__}, scikit-learn {sklearn.__version__}", "",
             "--- segment length x overlap (final pipeline otherwise; marked [FINAL] = main.py setting) ---",
             f"{'length(s)':>9}{'overlap':>9}{'samples/subj':>14}" + "".join(f"{m:>14}" for m in MODELS)]
    final_seg = (Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO)
    for L, ov in [(3, 0.5), (5, 0.5), (5, 0.75), (8, 0.5), (8, 0.75), (10, 0.5), (10, 0.75)]:
        Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO = L, ov
        n, acc = evaluate()
        line = f"{L:>9}{ov:>9.2f}{n:>14}" + "".join(f"{a:14.3f}" for a in acc) + (" [FINAL]" if (L, ov) == final_seg else "")
        lines.append(line); print(line, flush=True)
    Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO = final_seg

    lines += ["", "--- feature selection (SelectKBest, k features kept out of the final feature set) ---",
              f"{'k':>9}" + "".join(f"{m:>14}" for m in MODELS)]
    Config.FEATURE_SELECTION = True
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, s = main.load_all_subjects()
    for k in [8, 12, 15, 20, 25, 32]:
        Config.N_FEATURES_SELECT = k
        acc = [np.mean(main.leave_one_subject_out_validation(m, X, y, s)["accuracies"]) for m in MODELS]
        line = f"{k:>9}" + "".join(f"{a:14.3f}" for a in acc) + (" (= no selection)" if k == X.shape[1] else "")
        lines.append(line); print(line, flush=True)
    Config.FEATURE_SELECTION = False

    out = os.path.join(HERE, "results_pipeline.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main_exp()
