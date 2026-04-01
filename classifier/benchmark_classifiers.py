"""
Benchmark LR vs LinearSVC vs SVC(linear) on themes.json using 5-fold CV.
No model files are written — CV only.

Usage:
    pdm run python classifier/benchmark_classifiers.py
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC


def load_dataset(themes_path: str) -> tuple[list[str], list[str], list[str]]:
    with open(themes_path) as f:
        themes = json.load(f)
    texts, labels, theme_names = [], [], []
    for theme in themes:
        theme_names.append(theme["theme"])
        for example in theme["examples"]:
            texts.append(example)
            labels.append(theme["theme"])
    return texts, labels, theme_names


def run_cv(X: np.ndarray, y: np.ndarray, clf, theme_names: list[str], le: LabelEncoder) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    with open("data/themes.json") as f:
        label_to_theme = {t["label"]: t["theme"] for t in json.load(f)}
    display_names = [label_to_theme.get(c, c) for c in le.classes_]
    report = classification_report(y, y_pred, target_names=display_names, output_dict=True, zero_division=0)
    return report


def main() -> None:
    print("Loading dataset...")
    texts, labels, theme_names = load_dataset("data/themes.json")
    print(f"  {len(texts)} examples, {len(set(labels))} themes\n")

    print("Loading BAAI/bge-m3 and encoding...")
    model = SentenceTransformer("BAAI/bge-m3")
    X = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    le = LabelEncoder()
    # Use label slugs for encoding (consistent with train.py)
    with open("data/themes.json") as f:
        themes_data = json.load(f)
    theme_to_label = {t["theme"]: t["label"] for t in themes_data}
    raw_labels = [theme_to_label[l] for l in labels]
    y = le.fit_transform(raw_labels)

    classifiers = {
        "LR (current)": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        "LinearSVC + calibration": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, class_weight="balanced"), cv=5
        ),
        "SVC linear + Platt": SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"Running 5-fold CV: {name}...")
        report = run_cv(X, y, clf, theme_names, le)
        results[name] = report
        acc = report["accuracy"]
        f1 = report["macro avg"]["f1-score"]
        print(f"  Accuracy: {acc:.3f}  Macro F1: {f1:.3f}\n")

    # Per-class breakdown
    print("── Per-class F1 comparison")
    header = f"{'Theme':<25}" + "".join(f"  {n:<22}" for n in classifiers)
    print(header)
    print("-" * len(header))
    for theme in theme_names:
        row = f"{theme:<25}"
        for name, report in results.items():
            f1 = report.get(theme, {}).get("f1-score", 0.0)
            row += f"  {f1:.2f}{'':20}"
        print(row)

    print("\n── Summary")
    best_name = max(results, key=lambda n: results[n]["macro avg"]["f1-score"])
    for name, report in results.items():
        marker = " ← best" if name == best_name else ""
        print(f"  {name:<30}  acc={report['accuracy']:.3f}  macro F1={report['macro avg']['f1-score']:.3f}{marker}")


if __name__ == "__main__":
    main()
