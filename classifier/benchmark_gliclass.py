"""
Benchmark GLiClass (zero-shot) vs our trained LR head on themes.json.

GLiClass is NOT a project dependency — install it manually before running:
    pip install gliclass

Usage:
    pdm run python classifier/benchmark_gliclass.py
    pdm run python classifier/benchmark_gliclass.py --model knowledgator/gliclass-small-v1.0
    pdm run python classifier/benchmark_gliclass.py --device cuda  # if GPU available

Note on fairness: our LR head was *trained* on themes.json (5-fold CV result).
GLiClass runs zero-shot on the same data — no training, no examples.
The comparison answers: "does a trained head on 163 examples beat zero-shot?"
"""
import argparse
import json

from sklearn.metrics import accuracy_score, classification_report


def load_themes(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_dataset(themes: list[dict]) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for theme in themes:
        for example in theme["examples"]:
            texts.append(example)
            labels.append(theme["theme"])
    return texts, labels


def run_gliclass(texts: list[str], theme_names: list[str], model_name: str, device: str) -> list[str]:
    try:
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "gliclass is not installed. Run: pip install gliclass\n"
            "(It is not a project dependency — install it manually for this benchmark.)"
        )

    print(f"Loading {model_name}...")
    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = ZeroShotClassificationPipeline(
        model, tokenizer,
        classification_type="multi-label",
        device=device,
    )

    predictions = []
    for i, text in enumerate(texts):
        results = pipeline(text, theme_names, threshold=0.0)[0]
        # Take highest-scoring label regardless of threshold
        top = max(results, key=lambda r: r["score"])["label"] if results else "Autres"
        predictions.append(top)
        if (i + 1) % 25 == 0 or (i + 1) == len(texts):
            print(f"  {i + 1}/{len(texts)}")

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GLiClass vs LR head on themes.json")
    parser.add_argument("--themes", default="data/themes.json")
    parser.add_argument("--eval", default="data/classifier_eval.json")
    parser.add_argument("--model", default="knowledgator/gliclass-large-v1.0",
                        help="GLiClass model to benchmark")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    themes = load_themes(args.themes)
    texts, true_labels = build_dataset(themes)
    theme_names = [t["theme"] for t in themes]

    print(f"\nDataset: {len(texts)} examples, {len(themes)} themes\n")

    # Our head — use precomputed CV results
    with open(args.eval) as f:
        eval_data = json.load(f)

    print("── LR head (5-fold CV on same data, trained)")
    print(f"   Accuracy : {eval_data['accuracy']:.3f}")
    print(f"   Macro F1 : {eval_data['macro_f1']:.3f}")
    for name, stats in eval_data["per_class"].items():
        print(f"   {name:<25} F1={stats['f1']:.2f}  support={stats['support']}")

    # GLiClass zero-shot
    print(f"\n── GLiClass zero-shot ({args.model})")
    predictions = run_gliclass(texts, theme_names, args.model, args.device)

    acc = accuracy_score(true_labels, predictions)
    report = classification_report(
        true_labels, predictions,
        labels=theme_names,
        zero_division=0,
        output_dict=True,
    )
    print(f"   Accuracy : {acc:.3f}")
    print(f"   Macro F1 : {report['macro avg']['f1-score']:.3f}")
    for name in theme_names:
        stats = report.get(name, {})
        f1 = stats.get("f1-score", 0.0)
        sup = stats.get("support", 0)
        print(f"   {name:<25} F1={f1:.2f}  support={sup}")

    # Summary
    delta = acc - eval_data["accuracy"]
    winner = "GLiClass" if delta > 0 else "LR head"
    print(f"\n── Result: {winner} wins  (delta {delta:+.3f})")


if __name__ == "__main__":
    main()
