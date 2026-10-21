"""Check the collected data: 30 files per subject, 10240 lines per file, correct file names.

Usage:  python check_data.py [dataset folder, default bci_dataset_115-1]
"""
import os, re, sys
import numpy as np

EXPECTED_LINES = 10240
root = sys.argv[1] if len(sys.argv) > 1 else "bci_dataset_115-1"
ok_all = True
subjects = sorted(d.name for d in os.scandir(root) if d.is_dir())
print(f"Found {len(subjects)} subject(s) in {root}: {subjects}")
for sid in subjects:
    folder = os.path.join(root, sid)
    problems = []
    for task in (1, 2, 3):
        for rnd in range(1, 11):
            name = f"{sid}_{task}_{rnd}.txt"
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                problems.append(f"missing file {name}")
                continue
            try:
                x = np.loadtxt(path)
            except Exception as e:
                problems.append(f"{name}: failed to read ({e})")
                continue
            if x.ndim != 1 or len(x) != EXPECTED_LINES:
                problems.append(f"{name}: {len(x)} lines (expected {EXPECTED_LINES})")
            elif np.ptp(x) == 0:
                problems.append(f"{name}: signal is constant (headset may not be connected properly)")
    extra = [f for f in os.listdir(folder)
             if not re.fullmatch(re.escape(sid) + r"_[123]_(10|[1-9])\.txt", f)]
    if extra:
        problems.append(f"extra or wrongly named files: {extra}")
    if sid != sid.lower():
        problems.append("folder name should be the lowercase student ID")
    print(f"[{sid}] " + ("OK, all 30 files are fine" if not problems else "problems found:"))
    for p in problems:
        print("   -", p)
    ok_all &= not problems
print("\nAll checks passed" if ok_all else "\nPlease fix the problems above (files with too few lines must be re-recorded)")
