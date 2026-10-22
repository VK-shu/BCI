"""
Brain-Computer Interface Multi-Classifier (BrainLink Version)
For EEG signal Relax / Focus / Stroop state classification (Three-Class)
Compares 4 models: MLP, SVM, Random Forest, and a Self-Selected Model

Evaluation: Leave-One-Subject-Out (LOSO) cross-validation.
Outputs:
  - bci_multi_model_comparison.png : model comparison, per-subject accuracy,
                                     confusion matrix (counts + row-normalized)
  - results_summary.txt            : text summary, ready to paste into the report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
# Models
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# === STUDENT MODEL IMPORT HERE ===
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, welch
from scipy.stats import kurtosis
# =================================
from sklearn.feature_selection import SelectKBest, f_classif
import os
import glob
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"[{desc}] Processing...")
        return iterable

# Parameter Settings
class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_115-1"

    # Class settings (label 0 = Relax, 1 = Focus, 2 = Stroop)
    CLASS_NAMES = ['Relax', 'Focus', 'Stroop']

    # MLP parameters
    HIDDEN_LAYERS = (128, 64, 32)  # hhhy
    MAX_ITER = 150                 # hhhy
    LEARNING_RATE = 0.0025         # hhhy
    ALPHA = 0.005                  # hhhy

    # SVM parameters
    SVM_KERNEL = 'rbf'   # hhhy
    SVM_C = 1.0          # hhhy
    # gamma is left at the scikit-learn default ('scale'); the SVC constructor is not edited

    # Random Forest parameters
    RF_ESTIMATORS = 100  # hhhy
    RF_MAX_DEPTH = None  # hhhy

    # Signal processing parameters
    SAMPLING_RATE = 512    # BrainLink fixed sampling rate
    SEGMENT_LENGTH = 8     # Segment length in seconds      # hhhy
    OVERLAP_RATIO = 0.75   # Overlap ratio for segments     # hhhy

    # Preprocessing parameters (student)
    BANDPASS = (1.0, 45.0)     # Hz, band-pass range   # hhhy
    NOTCH_FREQ = 60.0          # Hz, mains frequency in Taiwan

    # Ablation switches (student): used by ablation.py to build Table 3 of the report.
    # Keep all three True for the final pipeline.
    USE_FILTERING = True       # DC removal + notch + band-pass
    USE_FREQ_FEATURES = True   # log band powers
    USE_TIME_FEATURES = True   # per-window log std + kurtosis
    SUBJECT_ZSCORE = True      # z-score every subject's features with that subject's own mean/std (no labels)
    STAT_WIN = 1.0             # seconds, short window inside a segment
    STAT_STEP = 0.25           # seconds, hop between short windows
    # If both feature groups are False, the (optionally filtered) raw samples are the features.

    # Feature parameters (student)
    BANDS = {                  # Hz; same six bands as in the lecture slides
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'lowbeta': (13, 20), 'highbeta': (20, 30), 'gamma': (30, 45),
    }

    # Feature selection parameters
    FEATURE_SELECTION = False  # hhhy
    N_FEATURES_SELECT = 20     # hhhy

    # Reporting settings
    TARGET_ACCURACY = 0.75   # required level
    FIGURE_NAME = 'bci_multi_model_comparison.png'
    SUMMARY_NAME = 'results_summary.txt'

    # Plot style: one fixed color per model, so that a model keeps the same
    # color in every panel even if one of the models is skipped.
    MODEL_COLORS = {
        'MLP': '#2a78d6',           # blue
        'SVM': '#eb6834',           # orange
        'RF': '#1baf7a',            # aqua
        'SelfSelected': '#4a3aa7',  # violet
    }
    FALLBACK_COLOR = '#8a8985'
    REFERENCE_COLOR = '#52514e'     # target / bonus reference lines
    HEATMAP_CMAP = 'Blues'          # sequential, single hue

    # Other settings
    RANDOM_STATE = 42

def create_segments(data, segment_length_samples, overlap_samples):
    """Split a single round of continuous EEG signal into multiple segments"""
    if len(data) < segment_length_samples:
        return []

    segments = []
    start = 0
    step = segment_length_samples - overlap_samples

    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]

        # ==========================================
        # === STUDENT PREPROCESSING HERE (Part 1)===
        # ==========================================
        if Config.USE_FILTERING:
            # 1) remove DC -> 60 Hz notch -> 1-45 Hz band-pass (zero-phase)
            segment = np.asarray(segment, dtype=float)
            segment = segment - np.mean(segment)
            b_notch, a_notch = iirnotch(Config.NOTCH_FREQ, Q=30, fs=Config.SAMPLING_RATE)
            segment = filtfilt(b_notch, a_notch, segment)
            sos_bp = butter(4, Config.BANDPASS, btype='bandpass', fs=Config.SAMPLING_RATE, output='sos')
            segment = sosfiltfilt(sos_bp, segment)

        segments.append(segment)
        start += step

    return segments

def extract_features(segments):
    """
    Perform feature engineering on segments
    """
    features = []
    for seg in segments:
        # ==========================================
        # === STUDENT PREPROCESSING HERE (Part 2)===
        # ==========================================
        if not (Config.USE_FREQ_FEATURES or Config.USE_TIME_FEATURES):
            features.append(seg)      # raw time-domain samples (ablation baseline)
            continue

        fs = Config.SAMPLING_RATE

        def segment_features(s):
            """Distribution (mean / std / 90th percentile / max) of 1 s window features inside one segment.
            A single Welch spectrum averages short alpha / theta bursts away; the distribution keeps them."""
            n = int(Config.STAT_WIN * fs)
            hop = max(1, int(Config.STAT_STEP * fs))
            rows = []
            for k in range(0, len(s) - n + 1, hop):
                w = s[k:k + n]
                row = []
                if Config.USE_FREQ_FEATURES:
                    freqs, psd = welch(w, fs=fs, nperseg=n, noverlap=n // 2)
                    df = freqs[1] - freqs[0]
                    bp = {name: psd[(freqs >= lo) & (freqs < hi)].sum() * df
                          for name, (lo, hi) in Config.BANDS.items()}
                    total = psd[(freqs >= 1) & (freqs < 45)].sum() * df + 1e-12
                    beta = bp['lowbeta'] + bp['highbeta']
                    row += [bp['alpha'] / total, bp['theta'] / total,
                            bp['lowbeta'] / total, bp['highbeta'] / total,
                            np.log10((bp['theta'] + 1e-12) / (beta + 1e-12)),    # theta / beta
                            np.log10((bp['alpha'] + 1e-12) / (beta + 1e-12))]    # alpha / beta
                if Config.USE_TIME_FEATURES:
                    row += [np.log10(np.std(w) + 1e-12), kurtosis(w)]           # amplitude, spikiness
                rows.append(row)
            series = np.asarray(rows, dtype=float)
            return np.concatenate([series.mean(axis=0), series.std(axis=0),
                                   np.percentile(series, 90, axis=0), series.max(axis=0)])

        # Subject-wise z-score. extract_features only receives segments (no subject id), so the
        # table {segment -> normalised feature vector} is built once per configuration by running
        # Part 1 + the features above on every recording and normalising each subject with that
        # subject's own mean / std (labels are never used). All four models share these features.
        cache_key = (Config.DATASET_PATH, Config.SEGMENT_LENGTH, Config.OVERLAP_RATIO,
                     Config.BANDPASS, Config.USE_FILTERING,
                     Config.USE_FREQ_FEATURES, Config.USE_TIME_FEATURES, Config.SUBJECT_ZSCORE,
                     Config.STAT_WIN, Config.STAT_STEP)
        if getattr(extract_features, "_key", None) != cache_key:
            seg_len = int(Config.SEGMENT_LENGTH * fs)
            ovl = int(seg_len * Config.OVERLAP_RATIO)
            per_subject = {}
            for path in sorted(glob.glob(os.path.join(Config.DATASET_PATH, "*", "*.txt"))):
                sid = os.path.basename(os.path.dirname(path))
                for s in create_segments(np.loadtxt(path), seg_len, ovl):
                    per_subject.setdefault(sid, []).append((np.asarray(s).tobytes(), segment_features(s)))
            table = {}
            for sid, items in per_subject.items():
                mat = np.vstack([v for _, v in items])
                mu, sd = mat.mean(axis=0), mat.std(axis=0)
                sd = np.where(sd < 1e-8, 1.0, sd)
                for key, v in items:
                    table[key] = (v - mu) / sd if Config.SUBJECT_ZSCORE else v
            extract_features._table, extract_features._key = table, cache_key

        current_feature = extract_features._table[np.asarray(seg).tobytes()]
        features.append(current_feature)

    return np.array(features)

def load_all_subjects():
    """Load round-based data for all subjects in the group"""
    all_features = []
    all_labels = []
    all_subjects = []

    if not os.path.exists(Config.DATASET_PATH):
        print(f"Error: Directory '{Config.DATASET_PATH}' not found")
        return None, None, None

    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])

    if len(subject_folders) < 2:
        print("Error: Not enough subjects. At least 2 subject folders are required for cross-validation.")
        return None, None, None

    print(f"Found {len(subject_folders)} subjects. Loading data...")

    segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
    overlap_samples = int(segment_length_samples * Config.OVERLAP_RATIO)

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)

        # segments_by_class[k] holds every segment of class k for this subject
        segments_by_class = [[] for _ in Config.CLASS_NAMES]

        for class_idx, class_name in enumerate(Config.CLASS_NAMES):
            # task number in the file name is class_idx + 1 (_1_ / _2_ / _3_)
            pattern = os.path.join(subject_folder, f"*_{class_idx + 1}_*.txt")
            for file in sorted(glob.glob(pattern)):
                try:
                    data = np.loadtxt(file)
                    segs = create_segments(data, segment_length_samples, overlap_samples)
                    segments_by_class[class_idx].extend(segs)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

        if any(len(segs) == 0 for segs in segments_by_class):
            print(f"Warning: Insufficient data for {subject_id}. Skipping.")
            continue

        # Extract features and build labels (0 = Relax, 1 = Focus, 2 = Stroop)
        subject_features = []
        subject_labels = []
        for class_idx, segs in enumerate(segments_by_class):
            feats = extract_features(segs)
            subject_features.append(feats)
            subject_labels.append(np.full(len(feats), class_idx))

        subject_features = np.vstack(subject_features)
        subject_labels = np.hstack(subject_labels)

        all_features.append(subject_features)
        all_labels.append(subject_labels)
        all_subjects.extend([subject_id] * len(subject_labels))

        counts = ", ".join(f"{len(segs)} {name}"
                           for name, segs in zip(Config.CLASS_NAMES, segments_by_class))
        print(f" - {subject_id}: Successfully loaded {counts} segments")

    if not all_features:
        return None, None, None

    return np.vstack(all_features), np.hstack(all_labels), all_subjects


class MultiModelBCIClassifier:
    """Wrapper class supporting multiple classifier evaluations"""
    def __init__(self, model_type='MLP'):
        self.model_type = model_type

        # Model Selection
        if model_type == 'MLP':
            self.model = MLPClassifier(
                hidden_layer_sizes=Config.HIDDEN_LAYERS,
                max_iter=Config.MAX_ITER,
                learning_rate_init=Config.LEARNING_RATE,
                alpha=Config.ALPHA,
                activation='relu',
                solver='adam',
                batch_size=32,
                early_stopping=True,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == 'SVM':
            self.model = SVC(
                kernel=Config.SVM_KERNEL,
                C=Config.SVM_C,
                probability=True,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == 'RF':
            self.model = RandomForestClassifier(
                n_estimators=Config.RF_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == 'SelfSelected':
            # ==========================================
            # === STUDENT MODEL HERE ===================
            # ==========================================
            # Shrinkage LDA: linear, few parameters, stable with small samples; a classic EEG/BCI baseline
            self.model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')  # hhhy
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=Config.N_FEATURES_SELECT) if Config.FEATURE_SELECTION else None

    def fit(self, X, y):
        if self.model is None:
            raise NotImplementedError("Model is currently blank (None).")

        X_scaled = self.scaler.fit_transform(X)
        if self.feature_selector is not None:
            self.feature_selector.k = min(Config.N_FEATURES_SELECT, X_scaled.shape[1])
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled

        self.model.fit(X_selected, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise NotImplementedError("Model is currently blank (None).")

        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # ==========================================
        # === STUDENT POSTPROCESSING HERE ==========
        # ==========================================

        return self.model.predict(X_selected)


def leave_one_subject_out_validation(model_type, X, y, subjects):
    """Run LOSO cross-validation for one model on pre-loaded features."""
    n_classes = len(Config.CLASS_NAMES)
    unique_subjects = sorted(set(subjects))
    results = {'accuracies': [], 'confusion_matrices': [], 'subject_names': []}

    for test_subject in tqdm(unique_subjects, desc=f"LOSO ({model_type:12})", leave=False):
        train_mask = [s != test_subject for s in subjects]
        test_mask = [s == test_subject for s in subjects]

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        classifier = MultiModelBCIClassifier(model_type=model_type)
        try:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        except NotImplementedError:
            return None

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

        results['accuracies'].append(accuracy)
        results['confusion_matrices'].append(cm)
        results['subject_names'].append(test_subject)

    return results


# ==============================================================
#  Reporting: accuracy summary + confusion matrix
# ==============================================================

def per_class_metrics(cm):
    """
    Return recall, precision and F1 for every class from a confusion matrix.
    Rows of cm are the true labels, columns are the predicted labels, so
      recall_k    = cm[k, k] / sum(cm[k, :])
      precision_k = cm[k, k] / sum(cm[:, k])
    """
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)      # number of true samples of each class
    predicted = cm.sum(axis=0)    # number of samples predicted as each class

    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    return recall, precision, f1, support.astype(int), predicted.astype(int)


def format_summary(all_results, best_model_name):
    """Build the text summary (also written to results_summary.txt)."""
    lines = []
    lines.append("=" * 50)
    lines.append("Summary of Classification Results (LOSO):")
    for model_type, res in all_results.items():
        mean_acc = np.mean(res['accuracies'])
        std_acc = np.std(res['accuracies'])
        lines.append(f" - {model_type:12}: {mean_acc:.3f} +/- {std_acc:.3f}")

    best_results = all_results[best_model_name]
    best_acc = np.mean(best_results['accuracies'])
    lines.append("")
    lines.append(f"Best Performing Model: {best_model_name} with Accuracy {best_acc:.3f}")
    if best_acc >= Config.TARGET_ACCURACY:
        lines.append(f"  -> reached the {Config.TARGET_ACCURACY:.0%} target")
    else:
        lines.append(f"  -> below the {Config.TARGET_ACCURACY:.0%} target")

    # Accuracy of every model on every held-out subject
    lines.append("")
    lines.append("Accuracy per held-out subject:")
    subject_names = best_results['subject_names']
    header = f"{'Subject':<16}" + "".join(f"{m:>14}" for m in all_results)
    lines.append(header)
    for i, subject in enumerate(subject_names):
        row = f"{subject:<16}" + "".join(f"{all_results[m]['accuracies'][i]:>14.3f}"
                                         for m in all_results)
        lines.append(row)

    # Per-class report of the best model
    total_cm = np.sum(best_results['confusion_matrices'], axis=0)
    recall, precision, f1, support, predicted = per_class_metrics(total_cm)

    lines.append("")
    lines.append(f"Detailed Class Report for Best Model [{best_model_name}]:")
    for k, class_name in enumerate(Config.CLASS_NAMES):
        tp = int(total_cm[k, k])
        lines.append(f"[{class_name} Class]")
        lines.append(f"  - Accuracy (Recall): {recall[k]:.3f} ({tp}/{support[k]})")
        lines.append(f"  - Precision: {precision[k]:.3f} ({tp}/{predicted[k]})")
        lines.append(f"  - F1-score: {f1[k]:.3f}")
        lines.append("")

    # Confusion matrix of the best model
    lines.append(f"Overall Confusion Matrix [{best_model_name}] (rows = true, cols = predicted):")
    header = f"{'':<10}" + "".join(f"{name:>10}" for name in Config.CLASS_NAMES)
    lines.append(header)
    for k, class_name in enumerate(Config.CLASS_NAMES):
        row = f"{class_name:<10}" + "".join(f"{int(v):>10}" for v in total_cm[k])
        lines.append(row)

    return "\n".join(lines)


def plot_comparison_results(all_results, best_model_name):
    """
    all_results: dict mapping model_type to its LOSO validation results
    Produces a 2x2 figure:
      (a) mean accuracy of every model, with the LOSO standard deviation
      (b) per-subject accuracy of every model
      (c) confusion matrix of the best model, in counts
      (d) confusion matrix of the best model, row-normalized (per-class recall)
    """
    model_types = list(all_results.keys())
    mean_accuracies = [np.mean(all_results[m]['accuracies']) for m in model_types]
    std_accuracies = [np.std(all_results[m]['accuracies']) for m in model_types]
    # Color follows the model, not its rank, so the colors stay consistent
    # between the two panels and when a model is skipped.
    colors = [Config.MODEL_COLORS.get(m, Config.FALLBACK_COLOR) for m in model_types]

    best_results = all_results[best_model_name]
    subject_names = best_results['subject_names']
    class_names = Config.CLASS_NAMES

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BCI Model Classification Comparison - LOSO Results', fontsize=16)
    ax_models, ax_subjects, ax_cm, ax_cm_norm = axes.ravel()

    def style_axis(ax):
        """Recessive grid and axes, so that the bars carry the information."""
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#d9d8d3', linewidth=0.8)
        ax.xaxis.grid(False)
        ax.tick_params(length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#d9d8d3')

    # (a) Model comparison bar chart
    bars = ax_models.bar(model_types, mean_accuracies, yerr=std_accuracies,
                         width=0.6, color=colors,
                         error_kw=dict(ecolor=Config.REFERENCE_COLOR,
                                       capsize=4, lw=1.2))
    ax_models.set_title('Overall Mean Accuracy Comparison')
    ax_models.set_ylabel('Mean Accuracy')
    ax_models.set_ylim(0, 1)
    ax_models.axhline(y=Config.TARGET_ACCURACY, color=Config.REFERENCE_COLOR,
                      linestyle='--', linewidth=1.2,
                      label=f'Target: {Config.TARGET_ACCURACY:.2f}')
    ax_models.legend(frameon=False, fontsize=9)
    style_axis(ax_models)
    for bar, mean_acc, std_acc in zip(bars, mean_accuracies, std_accuracies):
        ax_models.text(bar.get_x() + bar.get_width() / 2.0, mean_acc + std_acc + 0.02,
                       f'{mean_acc:.3f}', ha='center', va='bottom',
                       fontweight='bold', color='#0b0b0b')

    # (b) Per-subject accuracy of every model (grouped bars)
    n_models = len(model_types)
    x = np.arange(len(subject_names))
    group_width = 0.78
    width = group_width / n_models
    for i, model_type in enumerate(model_types):
        offset = (i - (n_models - 1) / 2) * width
        ax_subjects.bar(x + offset, all_results[model_type]['accuracies'],
                        width * 0.92,  # small gap between adjacent bars
                        label=model_type, color=colors[i])
    ax_subjects.set_title('Accuracy by Held-out Subject (LOSO)')
    ax_subjects.set_ylabel('Accuracy')
    ax_subjects.set_xticks(x)
    ax_subjects.set_xticklabels(subject_names)
    ax_subjects.set_ylim(0, 1)
    ax_subjects.axhline(y=Config.TARGET_ACCURACY, color=Config.REFERENCE_COLOR,
                        linestyle='--', linewidth=1.2,
                        label=f'Target: {Config.TARGET_ACCURACY:.2f}')
    ax_subjects.legend(frameon=False, fontsize=9, ncol=n_models + 1,
                       loc='upper center', bbox_to_anchor=(0.5, -0.06))
    style_axis(ax_subjects)

    # (c) Confusion matrix of the best model, in counts
    total_cm = np.sum(best_results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap=Config.HEATMAP_CMAP, cbar=False,
                linewidths=2, linecolor='white',
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_title(f'Confusion Matrix - counts ({best_model_name})')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.tick_params(length=0)

    # (d) Confusion matrix of the best model, row-normalized
    row_sums = total_cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.divide(total_cm, row_sums,
                            out=np.zeros(total_cm.shape, dtype=float),
                            where=row_sums > 0)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=Config.HEATMAP_CMAP,
                vmin=0, vmax=1, cbar=False, linewidths=2, linecolor='white',
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm_norm)
    ax_cm_norm.set_title(f'Confusion Matrix - row-normalized ({best_model_name})')
    ax_cm_norm.set_xlabel('Predicted')
    ax_cm_norm.set_ylabel('Actual')
    ax_cm_norm.tick_params(length=0)

    plt.tight_layout()
    plt.savefig(Config.FIGURE_NAME, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("BCI EEG Classification - Evaluation")
    print("=" * 60)

    # Load the dataset once and reuse it for every model,
    # so that all models are compared on exactly the same features.
    X, y, subjects = load_all_subjects()
    if X is None:
        print("Validation aborted. Please check the dataset directory.")
        return
    print(f"Feature matrix: {X.shape[0]} segments x {X.shape[1]} features")

    models_to_evaluate = ['MLP', 'SVM', 'RF', 'SelfSelected']
    all_results = {}

    # Run Leave-One-Subject-Out validation
    for model_type in models_to_evaluate:
        print(f"\nRunning Cross-Validation for model: [{model_type}]...")
        results = leave_one_subject_out_validation(model_type, X, y, subjects)

        if results is None:
            if model_type == 'SelfSelected':
                print("[Info] 'SelfSelected' is blank/None. Skipping evaluation.")
                continue
            else:
                print(f"Validation failed for model {model_type}.")
                return

        all_results[model_type] = results
        mean_acc = np.mean(results['accuracies'])
        print(f"-> [{model_type}] Overall Mean Accuracy: {mean_acc:.3f}")

    if not all_results:
        print("No model was evaluated successfully.")
        return

    # Pick the best model by mean LOSO accuracy
    best_model_name = max(all_results,
                          key=lambda m: np.mean(all_results[m]['accuracies']))

    # Text summary (printed and saved for the report)
    summary = format_summary(all_results, best_model_name)
    print("\n" + summary)
    with open(Config.SUMMARY_NAME, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")

    # Save comparison charts
    plot_comparison_results(all_results, best_model_name)
    print(f"\nResults saved in '{Config.FIGURE_NAME}' and '{Config.SUMMARY_NAME}'")


if __name__ == "__main__":
    main()
