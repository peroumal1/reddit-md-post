"""
Benchmark embedding models for thematic classification on themes.json.

Holds the classifier fixed (LinearSVC + CalibratedClassifierCV — our current best)
and varies the embedding backbone. 5-fold stratified CV, no model files written.

Variants tested:
  - BAAI/bge-m3                          (multilingual, also used for dedup)
  - intfloat/multilingual-e5-large-instruct  (instruct prompt)
  - bge-m3 + e5-instruct concatenated    (2048-dim — current production backbone)
  - Qwen/Qwen3-Embedding-0.6B            (instruct prompt, 2026 MTEB multilingual leader family)
  - bge-m3 + Qwen3-0.6B concatenated     (2048-dim candidate replacement)

Earlier runs also tested solon-embeddings-large and an e5 domain-specific prompt;
both lost to the current concat and were dropped from the matrix.

Usage:
    pdm run python classifier/benchmark_embeddings.py
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from rss_summary.classification import BGE_MODEL_ID, E5_MODEL_ID, E5_PROMPT

QWEN3_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def load_dataset(themes_path: str):
    with open(themes_path) as f:
        themes = json.load(f)
    texts, labels, theme_names, label_to_theme = [], [], [], {}
    for theme in themes:
        theme_names.append(theme["theme"])
        label_to_theme[theme["label"]] = theme["theme"]
        for example in theme["examples"]:
            texts.append(example)
            labels.append(theme["label"])
    return texts, labels, theme_names, label_to_theme


def encode_model(model_name: str, texts: list[str], prompt: str | None = None) -> np.ndarray:
    model = SentenceTransformer(model_name)
    if prompt:
        inputs = [prompt + t for t in texts]
    else:
        inputs = texts
    return model.encode(inputs, show_progress_bar=True, normalize_embeddings=True)


def run_cv(X: np.ndarray, y: np.ndarray, display_names: list[str]) -> dict:
    clf = CalibratedClassifierCV(
        LinearSVC(max_iter=2000, C=1.0, class_weight="balanced"), cv=5
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    return classification_report(y, y_pred, target_names=display_names, output_dict=True, zero_division=0)


def main() -> None:
    texts, raw_labels, theme_names, label_to_theme = load_dataset("data/themes.json")
    print(f"Dataset: {len(texts)} examples, {len(theme_names)} themes\n")

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)
    display_names = [label_to_theme[c] for c in le.classes_]

    # --- encode each backbone once, reuse for variants ---
    print(f"── Loading {BGE_MODEL_ID}...")
    X_bge = encode_model(BGE_MODEL_ID, texts)
    print(f"  shape: {X_bge.shape}\n")

    print(f"── Loading {E5_MODEL_ID}...")
    X_e5 = encode_model(E5_MODEL_ID, texts, prompt=E5_PROMPT)
    print(f"  shape: {X_e5.shape}\n")

    print(f"── Loading {QWEN3_MODEL_ID}...")
    X_qwen = encode_model(QWEN3_MODEL_ID, texts, prompt=E5_PROMPT)
    print(f"  shape: {X_qwen.shape}\n")

    variants = {
        "bge-m3": X_bge,
        "e5-instruct": X_e5,
        "qwen3-0.6b": X_qwen,
        "bge-m3 + e5 concat (current)": np.concatenate([X_bge, X_e5], axis=1),
        "bge-m3 + qwen3 concat": np.concatenate([X_bge, X_qwen], axis=1),
    }

    results = {}
    for name, X in variants.items():
        print(f"── CV: {name}...")
        report = run_cv(X, y, display_names)
        results[name] = report
        print(f"  Accuracy: {report['accuracy']:.3f}  Macro F1: {report['macro avg']['f1-score']:.3f}\n")

    # Per-class breakdown
    print("── Per-class F1")
    col_w = 28
    header = f"{'Theme':<25}" + "".join(f"  {n:<{col_w}}" for n in results)
    print(header)
    print("-" * len(header))
    for theme in theme_names:
        row = f"{theme:<25}"
        for report in results.values():
            f1 = report.get(theme, {}).get("f1-score", 0.0)
            row += f"  {f1:.2f}{'':<{col_w - 4}}"
        print(row)

    print("\n── Summary")
    baseline = "bge-m3 + e5 concat (current)"
    baseline_f1 = results[baseline]["macro avg"]["f1-score"]
    for name, report in results.items():
        acc = report["accuracy"]
        f1 = report["macro avg"]["f1-score"]
        delta = f"  ({f1 - baseline_f1:+.3f})" if name != baseline else "  (baseline)"
        marker = " ← best" if f1 == max(r["macro avg"]["f1-score"] for r in results.values()) else ""
        print(f"  {name:<35}  acc={acc:.3f}  macro F1={f1:.3f}{delta}{marker}")


if __name__ == "__main__":
    main()
