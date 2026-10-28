"""Hyper-parameter experiments for the Random Forest (RandomForestClassifier).

One parameter is varied at a time while all others stay at the final value used in main.py;
the framework's initial values (main_original.py: 100 trees, max_depth None) are marked [INIT].
Usage (from the project root):   uv run python exp_testing/exp_rf.py
"""
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

# Run from the project root so that main.py and the dataset folder are found.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import main                      # noqa: E402  (imports only; main.py is not modified or run)
from main import Config          # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = None                  # filled by load_features()


def load_features():
    """Final feature matrix of main.py (same preprocessing / features / subject z-score)."""
    global SUBJECTS
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, subjects = main.load_all_subjects()
    SUBJECTS = sorted(set(subjects))
    return X, y, np.array(subjects)


def loso(make_model, X, y, subjects):
    """Leave-one-subject-out accuracy per held-out subject (StandardScaler fitted on the training part,
    exactly like MultiModelBCIClassifier with FEATURE_SELECTION = False)."""
    accs = []
    for u in SUBJECTS:
        tr, te = subjects != u, subjects == u
        scaler = StandardScaler().fit(X[tr])
        model = make_model().fit(scaler.transform(X[tr]), y[tr])
        accs.append(float((model.predict(scaler.transform(X[te])) == y[te]).mean()))
    return np.array(accs)


class Report:
    """Collects the sweep rows, prints them and saves them to a text file next to this script."""

    def __init__(self, title, out_name):
        self.out = os.path.join(HERE, out_name)
        import sklearn
        self.lines = [title, "=" * len(title),
                      "LOSO accuracy (3 held-out subjects); marks: [INIT] = framework's initial value (main_original.py),",
                      "[FINAL] = value used in main.py. All other parameters stay at their final value.",
                      f"Reproducibility: random_state = {Config.RANDOM_STATE} unless a seed sweep is stated; "
                      f"python {sys.version.split()[0]}, numpy {np.__version__}, scikit-learn {sklearn.__version__}; "
                      f"segments {Config.SEGMENT_LENGTH}s / overlap {Config.OVERLAP_RATIO}.", ""]
        self._emit("\n".join(self.lines))
        self.lines = list(self.lines)

    def _emit(self, text):
        print(text, flush=True)

    def section(self, name):
        text = f"\n--- {name} ---\n{'setting':<34}{'mean':>8}{'std':>8}   per subject (" + " / ".join(s[-3:] for s in SUBJECTS) + ")"
        self.lines.append(text)
        self._emit(text)

    def run(self, label, make_model, X, y, subjects, init=False, final=False):
        t0 = time.time()
        try:
            a = loso(make_model, X, y, subjects)
            tag = (" [INIT]" if init else "") + (" [FINAL]" if final else "")
            text = (f"{str(label):<34}{a.mean():8.3f}{a.std():8.3f}   " + " / ".join(f"{v:.3f}" for v in a)
                    + f"{tag}")
            print(f"    [{time.time() - t0:.1f}s]", flush=True)
        except Exception as e:                      # an invalid combination should not stop the sweep
            text = f"{str(label):<34}  failed: {type(e).__name__}: {str(e)[:60]}"
        self.lines.append(text)
        self._emit(text)

    def run_seeds(self, label, factory, X, y, subjects, seeds, init=False, final=False):
        """Mean over several random seeds (for models whose result depends on random_state).
        mean = average of the seeds' LOSO means, std = std over the seeds, per subject = average over seeds."""
        accs = np.array([loso(lambda sd=sd: factory(sd), X, y, subjects) for sd in seeds])
        tag = (" [INIT]" if init else "") + (" [FINAL]" if final else "")
        text = (f"{str(label):<34}{accs.mean():8.3f}{accs.mean(1).std():8.3f}   "
                + " / ".join(f"{v:.3f}" for v in accs.mean(0)) + tag)
        self.lines.append(text)
        self._emit(text)
        return accs.mean()

    def top(self, rows, n=10):
        """rows = [(mean, label)]; appends the n best settings."""
        self.lines.append(f"top {n}:")
        self._emit(f"top {n}:")
        for m, lab in sorted(rows, key=lambda r: -r[0])[:n]:
            self.lines.append(f"  {m:.3f}  {lab}")
            self._emit(f"  {m:.3f}  {lab}")

    def save(self):
        with open(self.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        print(f"\nSaved to {self.out}")

from sklearn.ensemble import RandomForestClassifier

INIT = dict(n_estimators=100, max_depth=None)                              # main_original.py


def build(**kw):
    p = dict(n_estimators=Config.RF_ESTIMATORS, max_depth=Config.RF_MAX_DEPTH,
             random_state=Config.RANDOM_STATE, n_jobs=-1)
    p.update(kw)
    return lambda: RandomForestClassifier(**p)


def main_exp():
    X, y, s = load_features()
    r = Report("Random Forest hyper-parameter experiments", "results_rf.txt")

    r.section("final configuration (reference)")
    r.run("final setting", build(), X, y, s, final=True)

    r.section("n_estimators")
    for v in [5, 10, 25, 50, 100, 200, 300, 500, 1000]:
        r.run(v, build(n_estimators=v), X, y, s, init=v == INIT["n_estimators"], final=v == Config.RF_ESTIMATORS)

    r.section("max_depth")
    for v in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, None]:
        r.run(v, build(max_depth=v), X, y, s, init=v == INIT["max_depth"], final=v == Config.RF_MAX_DEPTH)

    r.section("min_samples_leaf (not in Config)")
    for v in [1, 2, 3, 5, 10, 20, 40]:
        r.run(v, build(min_samples_leaf=v), X, y, s, final=v == 1)

    r.section("min_samples_split (not in Config)")
    for v in [2, 5, 10, 20, 50]:
        r.run(v, build(min_samples_split=v), X, y, s, final=v == 2)

    r.section("max_features (not in Config)")
    for v in ["sqrt", "log2", 0.1, 0.25, 0.5, 0.75, None]:
        r.run(v, build(max_features=v), X, y, s, final=v == "sqrt")

    r.section("criterion (not in Config)")
    for v in ["gini", "entropy", "log_loss"]:
        r.run(v, build(criterion=v), X, y, s, final=v == "gini")

    r.section("bootstrap / max_samples (not in Config)")
    r.run("bootstrap=True", build(bootstrap=True), X, y, s, final=True)
    r.run("bootstrap=False", build(bootstrap=False), X, y, s)
    for v in [0.3, 0.5, 0.8]:
        r.run(f"max_samples={v}", build(max_samples=v), X, y, s)

    r.section("class_weight (not in Config)")
    for v in [None, "balanced", "balanced_subsample"]:
        r.run(v, build(class_weight=v), X, y, s, final=v is None)

    r.section("random_state (10 seeds; mean/std here are over the seeds' LOSO means)")
    means = []
    for seed in range(10):
        a = loso(build(random_state=seed), X, y, s)
        means.append(a.mean())
        line = f"seed {seed:<29}{a.mean():8.3f}        " + " / ".join(f"{v:.3f}" for v in a) + ("  [FINAL seed]" if seed == Config.RANDOM_STATE else "")
        r.lines.append(line); print(line, flush=True)
    line = f"{'mean over 10 seeds':<34}{np.mean(means):8.3f}{np.std(means):8.3f}"
    r.lines.append(line); print(line, flush=True)

    r.section("2-D grid: n_estimators x max_depth")
    for n in [50, 100, 300]:
        for d in [3, 6, None]:
            r.run(f"n={n} / depth={d}", build(n_estimators=n, max_depth=d), X, y, s)
    r.section("joint grid: n_estimators x max_depth (mean over seeds 0-4)")
    rows = []
    for n in [50, 100, 300, 500]:
        for d in [3, 4, 5, 6, 8, 10, None]:
            lab = f"n={n} / depth={d}"
            m = r.run_seeds(lab, lambda sd, n=n, d=d: build(n_estimators=n, max_depth=d, random_state=sd)(), X, y, s,
                            seeds=range(5), init=(n == INIT["n_estimators"] and d == INIT["max_depth"]),
                            final=(n == Config.RF_ESTIMATORS and d == Config.RF_MAX_DEPTH))
            rows.append((m, lab))
    r.top(rows, 10)
    r.save()


if __name__ == "__main__":
    main_exp()
