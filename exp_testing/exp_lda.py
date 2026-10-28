"""Experiments for the self-selected model (shrinkage LDA) and for alternative candidates.

Part A varies the LDA settings (solver, shrinkage, priors, tolerance); the final choice in main.py is
solver='lsqr', shrinkage='auto'. Part B compares other candidate classifiers that could have been the
fourth model (logistic regression, k-NN, naive Bayes, QDA, ridge, linear SVM, gradient boosting).
Usage (from the project root):   uv run python exp_testing/exp_lda.py
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def lda(**kw):
    p = dict(solver="lsqr", shrinkage="auto")
    p.update(kw)
    return lambda: LDA(**p)


def main_exp():
    X, y, s = load_features()
    r = Report("Self-selected model: shrinkage LDA settings and alternative candidates", "results_lda.txt")
    r.lines.append("(No framework initial value exists for the self-selected model, so only [FINAL] is marked.)")

    r.section("A1. final configuration (reference)")
    r.run("lsqr + shrinkage='auto'", lda(), X, y, s, final=True)

    r.section("A2. solver / shrinkage")
    r.run("svd (no shrinkage)", lambda: LDA(solver="svd"), X, y, s)
    r.run("lsqr, shrinkage=None", lambda: LDA(solver="lsqr", shrinkage=None), X, y, s)
    r.run("lsqr, shrinkage='auto'", lda(), X, y, s, final=True)
    r.run("eigen, shrinkage=None", lambda: LDA(solver="eigen", shrinkage=None), X, y, s)
    r.run("eigen, shrinkage='auto'", lambda: LDA(solver="eigen", shrinkage="auto"), X, y, s)

    r.section("A3. manual shrinkage value (lsqr; 0 = plain LDA, 1 = diagonal covariance)")
    for v in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r.run(f"shrinkage={v}", lda(shrinkage=v), X, y, s)

    r.section("A4. class priors")
    r.run("priors=None (from data)", lda(), X, y, s, final=True)
    r.run("priors=uniform (1/3 each)", lda(priors=[1 / 3] * 3), X, y, s)

    r.section("A5. svd solver tol / store_covariance (should not matter)")
    for v in [1e-6, 1e-4, 1e-2]:
        r.run(f"svd tol={v}", lambda v=v: LDA(solver="svd", tol=v), X, y, s)

    r.section("B1. QDA: regularization (reg_param)")
    for v in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        r.run(f"reg_param={v}", lambda v=v: QDA(reg_param=v), X, y, s)

    r.section("B2. logistic regression: C")
    for v in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        r.run(f"C={v}", lambda v=v: LogisticRegression(C=v, max_iter=2000), X, y, s)

    r.section("B3. k-NN: n_neighbors")
    for v in [1, 3, 5, 9, 15, 25, 41, 61]:
        r.run(f"k={v}", lambda v=v: KNeighborsClassifier(n_neighbors=v), X, y, s)
    r.run("k=15, distance-weighted", lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"), X, y, s)

    r.section("B4. other candidates")
    r.run("GaussianNB", lambda: GaussianNB(), X, y, s)
    for v in [1.0, 100.0, 1000.0]:
        r.run(f"RidgeClassifier alpha={v}", lambda v=v: RidgeClassifier(alpha=v), X, y, s)
    for v in [0.001, 0.01, 0.1]:
        r.run(f"LinearSVC C={v}", lambda v=v: LinearSVC(C=v, max_iter=20000, random_state=Config.RANDOM_STATE), X, y, s)
    r.run("ExtraTrees (300 trees)", lambda: ExtraTreesClassifier(n_estimators=300, random_state=Config.RANDOM_STATE, n_jobs=-1), X, y, s)
    r.run("GradientBoosting (100, depth 3)", lambda: GradientBoostingClassifier(random_state=Config.RANDOM_STATE), X, y, s)
    r.save()


if __name__ == "__main__":
    main_exp()
