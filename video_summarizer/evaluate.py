import json
from rouge_score import rouge_scorer

# Paths to files
GROUND_TRUTH_PATH = "results/summaries/ground_truth.json"
GENERATED_SUMMARY_PATH = "results/summaries/sample_summary.txt"


def load_ground_truth():
    """Loads ground truth summaries from the JSON file."""
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["output"] for item in data]  # Extract only the summary texts


def load_generated_summary():
    """Loads the generated summary from the text file."""
    with open(GENERATED_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = f.read().strip()
    return summary.split("\n")  # Split into individual summary points


def evaluate_summaries(ground_truths, generated_summaries):
    """Calculates ROUGE Precision, Recall, and F1-score."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for gt, gen in zip(ground_truths, generated_summaries):
        scores = scorer.score(gt, gen)

        rouge1_scores.append(scores["rouge1"])
        rouge2_scores.append(scores["rouge2"])
        rougeL_scores.append(scores["rougeL"])

    def avg_score(scores):
        """Computes the average precision, recall, and F1-score."""
        return {
            "Precision": sum([s.precision for s in scores]) / len(scores),
            "Recall": sum([s.recall for s in scores]) / len(scores),
            "F1-score": sum([s.fmeasure for s in scores]) / len(scores),
        }

    return {
        "ROUGE-1": avg_score(rouge1_scores),
        "ROUGE-2": avg_score(rouge2_scores),
        "ROUGE-L": avg_score(rougeL_scores),
    }


if __name__ == "__main__":
    # Load data
    ground_truth_summaries = load_ground_truth()
    generated_summaries = load_generated_summary()

    # Ensure equal lengths by trimming extra data
    min_len = min(len(ground_truth_summaries), len(generated_summaries))
    ground_truth_summaries = ground_truth_summaries[:min_len]
    generated_summaries = generated_summaries[:min_len]

    # Evaluate
    results = evaluate_summaries(ground_truth_summaries, generated_summaries)

    # Print results
    print("\n🔹 **Summary Evaluation Metrics:**")
    for metric, values in results.items():
        print(f"\n{metric}:")
        print(f"   Precision: {values['Precision']:.4f}")
        print(f"   Recall: {values['Recall']:.4f}")
        print(f"   F1-score: {values['F1-score']:.4f}")
