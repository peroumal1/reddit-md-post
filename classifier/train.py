"""
Train a LinearSVC classifier on concatenated BAAI/bge-m3 + e5-instruct embeddings.

Usage:
    pdm run python classifier/train.py
    pdm run python classifier/train.py --themes data/themes.json --output data/classifier_head.joblib

The trained head is ~1MB and is loaded by classification.py at inference time.
Both bge-m3 (already loaded for deduplication) and e5-instruct are used as frozen encoders.
Their 1024-dim embeddings are concatenated into a 2048-dim vector before classification.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from rss_summary.classification import E5_MODEL_ID, E5_PROMPT

BGE_MODEL_ID = "BAAI/bge-m3"


def _make_clf():
    return CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, class_weight="balanced"), cv=5)


def load_themes(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_dataset(themes: list[dict]) -> tuple[list[str], list[str]]:
    """Return (texts, labels) from themes.json examples."""
    texts, labels = [], []
    for theme in themes:
        for example in theme["examples"]:
            texts.append(example)
            labels.append(theme["label"])
    return texts, labels


def encode_concat(texts: list[str]) -> np.ndarray:
    """Encode texts with bge-m3 + e5-instruct, concatenate, and L2-normalize each row."""
    print(f"\nLoading {BGE_MODEL_ID}...")
    model_bge = SentenceTransformer(BGE_MODEL_ID)
    print(f"Encoding {len(texts)} examples with bge-m3...")
    X_bge = model_bge.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    print(f"\nLoading {E5_MODEL_ID}...")
    model_e5 = SentenceTransformer(E5_MODEL_ID)
    prefixed = [E5_PROMPT + t for t in texts]
    print(f"Encoding {len(texts)} examples with e5-instruct...")
    X_e5 = model_e5.encode(prefixed, show_progress_bar=True, normalize_embeddings=True)

    X = np.concatenate([X_bge, X_e5], axis=1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1)


def train(themes_path: str, output_path: str, eval_path: str) -> None:
    themes = load_themes(themes_path)
    texts, raw_labels = build_dataset(themes)

    print(f"Dataset: {len(texts)} examples across {len(themes)} themes")
    for theme in themes:
        count = raw_labels.count(theme["label"])
        print(f"  {theme['theme']}: {count} examples")

    X = encode_concat(texts)

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)

    # Cross-validation evaluation (more reliable than a single split given small data)
    print("\nRunning 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(_make_clf(), X, y, cv=cv)

    label_to_theme = {t["label"]: t["theme"] for t in themes}
    display_names = [label_to_theme[cls] for cls in le.classes_]

    report_str = classification_report(y, y_pred_cv, target_names=display_names)
    report_dict = classification_report(
        y, y_pred_cv, target_names=display_names, output_dict=True
    )
    print("\nCross-validation results:")
    print(report_str)

    print("Training final model on all examples...")
    t0 = time.time()
    clf = _make_clf()
    clf.fit(X, y)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    head = {
        "clf": clf,
        "label_encoder": le,
        "label_to_theme": label_to_theme,
        "theme_to_label": {t["theme"]: t["label"] for t in themes},
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, output_path)
    print(f"\nHead saved to {output_path}")

    eval_data = {
        "timestamp": datetime.now().isoformat(),
        "backbone": f"{BGE_MODEL_ID} + {E5_MODEL_ID} (concat 2048-dim)",
        "num_classes": len(themes),
        "num_examples": len(texts),
        "cv_folds": 5,
        "accuracy": report_dict["accuracy"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "per_class": {
            name: {
                "f1": report_dict[name]["f1-score"],
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "support": report_dict[name]["support"],
            }
            for name in display_names
        },
    }
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"Eval saved to {eval_path}")
    print(f"\nOverall accuracy: {eval_data['accuracy']:.3f}")
    print(f"Macro F1:         {eval_data['macro_f1']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier head on themes.json")
    parser.add_argument("--themes", default="data/themes.json")
    parser.add_argument("--output", default="data/classifier_head.joblib")
    parser.add_argument("--eval", default="data/classifier_eval.json")
    args = parser.parse_args()
    train(args.themes, args.output, args.eval)
