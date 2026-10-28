"""Hyper-parameter experiments for the MLP (MLPClassifier).

One parameter is varied at a time while all others stay at the final value used in main.py;
the framework's initial values (main_original.py) are included and marked [INIT].
Usage (from the project root):   uv run python exp_testing/exp_mlp.py
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

from sklearn.neural_network import MLPClassifier

INIT = dict(hidden=(128, 64, 32), max_iter=150, lr=0.002, alpha=0.005)      # main_original.py


def build(**kw):
    p = dict(hidden_layer_sizes=Config.HIDDEN_LAYERS, max_iter=Config.MAX_ITER,
             learning_rate_init=Config.LEARNING_RATE, alpha=Config.ALPHA, activation="relu",
             solver="adam", batch_size=32, early_stopping=True, random_state=Config.RANDOM_STATE)
    p.update(kw)
    return lambda: MLPClassifier(**p)


def main_exp():
    X, y, s = load_features()
    r = Report("MLP hyper-parameter experiments", "results_mlp.txt")
    fin = dict(hidden=Config.HIDDEN_LAYERS, max_iter=Config.MAX_ITER, lr=Config.LEARNING_RATE, alpha=Config.ALPHA)

    r.section("final configuration (reference)")
    r.run("final setting", build(), X, y, s, final=True)

    r.section("hidden_layer_sizes")
    for v in [(16,), (32,), (64,), (128,), (256,), (32, 16), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64),
              (64, 64, 64), (256, 128, 64, 32)]:
        r.run(v, build(hidden_layer_sizes=v), X, y, s, init=v == INIT["hidden"], final=v == fin["hidden"])

    r.section("learning_rate_init")
    for v in [0.0001, 0.0005, 0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.05]:
        r.run(v, build(learning_rate_init=v), X, y, s, init=v == INIT["lr"], final=v == fin["lr"])

    r.section("alpha (L2 penalty)")
    for v in [0.0, 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        r.run(v, build(alpha=v), X, y, s, init=v == INIT["alpha"], final=v == fin["alpha"])

    r.section("max_iter")
    for v in [10, 25, 50, 100, 150, 300, 500, 1000]:
        r.run(v, build(max_iter=v), X, y, s, init=v == INIT["max_iter"], final=v == fin["max_iter"])

    r.section("activation (not in Config)")
    for v in ["relu", "tanh", "logistic", "identity"]:
        r.run(v, build(activation=v), X, y, s, final=v == "relu")

    r.section("batch_size (not in Config)")
    for v in [8, 16, 32, 64, 128, 256]:
        r.run(v, build(batch_size=v), X, y, s, final=v == 32)

    r.section("solver (not in Config)")
    for v in ["adam", "sgd", "lbfgs"]:
        r.run(v, build(solver=v), X, y, s, final=v == "adam")

    r.section("early_stopping (not in Config)")
    for v in [True, False]:
        r.run(v, build(early_stopping=v), X, y, s, final=v is True)

    r.section("random_state (10 seeds; mean/std here are over the seeds' LOSO means)")
    means = []
    for seed in range(10):
        a = loso(build(random_state=seed), X, y, s)
        means.append(a.mean())
        line = f"seed {seed:<29}{a.mean():8.3f}        " + " / ".join(f"{v:.3f}" for v in a) + ("  [FINAL seed]" if seed == Config.RANDOM_STATE else "")
        r.lines.append(line); print(line, flush=True)
    line = f"{'mean over 10 seeds':<34}{np.mean(means):8.3f}{np.std(means):8.3f}"
    r.lines.append(line); print(line, flush=True)

    r.section("2-D grid: hidden_layer_sizes x alpha")
    for h in [(32,), (64, 32), (128, 64, 32)]:
        for a_ in [1e-4, 0.005, 0.1]:
            r.run(f"{h} / alpha={a_}", build(hidden_layer_sizes=h, alpha=a_), X, y, s)
    r.section("joint grid: hidden_layer_sizes x learning_rate_init x alpha (mean over seeds 0, 1, 2; max_iter = 150)")
    rows = []
    for h in [(32,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]:
        for lr in [0.0005, 0.001, 0.0025, 0.005, 0.01]:
            for al in [0.0001, 0.005, 0.05, 0.5, 2.0]:
                lab = f"{h} / lr={lr} / alpha={al}"
                m = r.run_seeds(lab, lambda sd, h=h, lr=lr, al=al: build(hidden_layer_sizes=h, learning_rate_init=lr, alpha=al, random_state=sd)(),
                                X, y, s, seeds=[0, 1, 2],
                                final=(h == fin["hidden"] and lr == fin["lr"] and al == fin["alpha"]))
                rows.append((m, lab))
    r.top(rows, 10)
    r.save()


if __name__ == "__main__":
    main_exp()
