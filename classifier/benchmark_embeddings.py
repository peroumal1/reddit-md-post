"""
Benchmark embedding models for thematic classification on themes.json.

Holds the classifier fixed (LinearSVC + CalibratedClassifierCV — our current best)
and varies the embedding backbone. 5-fold stratified CV, no model files written.

Variants tested:
  - BAAI/bge-m3                          (current baseline, multilingual)
  - intfloat/multilingual-e5-large-instruct  generic prompt
  - intfloat/multilingual-e5-large-instruct  domain-specific prompt (Caribbean news)
  - OrdalieTech/solon-embeddings-large-0.1   (French-specialized)
  - bge-m3 + e5-instruct concatenated    (2048-dim ensemble)

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

E5_MODEL = "intfloat/multilingual-e5-large-instruct"

PROMPT_GENERIC = (
    "Instruct: Classify the following French news headline into a thematic category.\nQuery: "
)
PROMPT_DOMAIN = (
    "Instruct: Classify this French Caribbean regional news article (from Guadeloupe or Martinique) "
    "into one of these categories: faits divers, politique, culture, sport, santé, économie, "
    "éducation, environnement, Outre-mer et Caraïbes, international.\nQuery: "
)


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
    print("── Loading BAAI/bge-m3...")
    X_bge = encode_model("BAAI/bge-m3", texts)
    print(f"  shape: {X_bge.shape}\n")

    print("── Loading intfloat/multilingual-e5-large-instruct...")
    X_e5_generic = encode_model(E5_MODEL, texts, prompt=PROMPT_GENERIC)
    print(f"  shape (generic prompt): {X_e5_generic.shape}")
    X_e5_domain = encode_model(E5_MODEL, texts, prompt=PROMPT_DOMAIN)
    print(f"  shape (domain prompt):  {X_e5_domain.shape}\n")

    print("── Loading OrdalieTech/solon-embeddings-large-0.1...")
    X_solon = encode_model("OrdalieTech/solon-embeddings-large-0.1", texts)
    print(f"  shape: {X_solon.shape}\n")

    # Concatenated ensemble
    X_concat = np.concatenate([X_bge, X_e5_domain], axis=1)

    variants = {
        "bge-m3 (baseline)": X_bge,
        "e5-instruct generic prompt": X_e5_generic,
        "e5-instruct domain prompt": X_e5_domain,
        "solon-embeddings-large": X_solon,
        "bge-m3 + e5 concat (2048-dim)": X_concat,
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
    baseline_f1 = results["bge-m3 (baseline)"]["macro avg"]["f1-score"]
    for name, report in results.items():
        acc = report["accuracy"]
        f1 = report["macro avg"]["f1-score"]
        delta = f"  ({f1 - baseline_f1:+.3f})" if name != "bge-m3 (baseline)" else "  (baseline)"
        marker = " ← best" if f1 == max(r["macro avg"]["f1-score"] for r in results.values()) else ""
        print(f"  {name:<35}  acc={acc:.3f}  macro F1={f1:.3f}{delta}{marker}")


if __name__ == "__main__":
    main()
