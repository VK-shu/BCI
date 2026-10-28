"""Hyper-parameter experiments for the SVM (SVC).

One parameter is varied at a time while all others stay at the final value used in main.py;
the framework's initial values (main_original.py: rbf kernel, C = 1.0) are marked [INIT].
Usage (from the project root):   uv run python exp_testing/exp_svm.py
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

from sklearn.svm import SVC

INIT = dict(kernel="rbf", C=1.0)                                            # main_original.py


def build(**kw):
    # probability=False does not change SVC.predict (only predict_proba) and is much faster.
    p = dict(kernel=Config.SVM_KERNEL, C=Config.SVM_C, gamma="scale", probability=False,
             random_state=Config.RANDOM_STATE)
    p.update(kw)
    return lambda: SVC(**p)


def main_exp():
    X, y, s = load_features()
    r = Report("SVM hyper-parameter experiments", "results_svm.txt")

    r.section("final configuration (reference)")
    r.run("final setting", build(), X, y, s, final=True)

    r.section("kernel")
    for v in ["linear", "rbf", "poly", "sigmoid"]:
        r.run(v, build(kernel=v), X, y, s, init=v == INIT["kernel"], final=v == Config.SVM_KERNEL)

    r.section("C (rbf kernel)")
    for v in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]:
        r.run(v, build(C=v), X, y, s, init=v == INIT["C"], final=v == Config.SVM_C)

    r.section("gamma (rbf kernel; 'scale' = 1/(n_features*X.var()) is the default and the final setting)")
    for v in ["scale", "auto", 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        r.run(v, build(gamma=v), X, y, s, final=v == "scale")

    r.section("C (linear kernel)")
    for v in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        r.run(v, build(kernel="linear", C=v), X, y, s)

    r.section("degree (poly kernel, C = final)")
    for v in [2, 3, 4, 5]:
        r.run(v, build(kernel="poly", degree=v), X, y, s)

    r.section("class_weight")
    for v in [None, "balanced"]:
        r.run(v, build(class_weight=v), X, y, s, final=v is None)

    r.section("shrinking / decision_function_shape")
    r.run("shrinking=False", build(shrinking=False), X, y, s)
    r.run("decision_function_shape='ovr'", build(decision_function_shape="ovr"), X, y, s)

    r.section("2-D grid: C x gamma (rbf)")
    for c in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        for g in ["scale", 0.003, 0.01, 0.03, 0.1]:
            r.run(f"C={c} / gamma={g}", build(C=c, gamma=g), X, y, s)
    r.section("joint grid: kernel x C")
    rows = []
    for k in ["rbf", "linear", "sigmoid", "poly"]:
        for c in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
            lab = f"{k} / C={c}"
            r.run(lab, build(kernel=k, C=c), X, y, s, init=(k == INIT["kernel"] and c == INIT["C"]),
                  final=(k == Config.SVM_KERNEL and c == Config.SVM_C))
            rows.append((loso(build(kernel=k, C=c), X, y, s).mean(), lab))
    r.top(rows, 10)
    r.save()


if __name__ == "__main__":
    main_exp()
