"""

import json
from rouge_score import rouge_scorer

# Paths to files
GROUND_TRUTH_PATH = "results/summaries/ground_truth.json"
GENERATED_SUMMARY_PATH = "results/summaries/sample_summary.txt"


def load_ground_truth():
    //Loads ground truth summaries from the JSON file.
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["output"] for item in data]  # Extract only the summary texts


def load_generated_summary():
    //Loads the generated summary from the text file.
    with open(GENERATED_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = f.read().strip()
    return summary.split("\n")  # Split into individual summary points


def evaluate_summaries(ground_truths, generated_summaries):
    //Calculates ROUGE Precision, Recall, and F1-score."
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for gt, gen in zip(ground_truths, generated_summaries):
        scores = scorer.score(gt, gen)

        rouge1_scores.append(scores["rouge1"])
        rouge2_scores.append(scores["rouge2"])
        rougeL_scores.append(scores["rougeL"])

    def avg_score(scores):
        //Computes the average precision, recall, and F1-score.
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

"""


"""

import json
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

# Paths to files
GROUND_TRUTH_PATH = "results/summaries/ground_truth.json"
GENERATED_SUMMARY_PATH = "results/summaries/sample_summary.txt"

def load_ground_truth():
    #Loads ground truth summaries from the JSON file.
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["output"] for item in data]  # Extract only the summary texts

def load_generated_summary():
    #Loads the generated summary from the text file.
    with open(GENERATED_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = f.read().strip()
    return summary.split("\n")  # Split into individual summary points

def evaluate_summaries(ground_truths, generated_summaries):
    #Calculates ROUGE Precision, Recall, and F1-score.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for gt, gen in zip(ground_truths, generated_summaries):
        scores = scorer.score(gt, gen)

        rouge1_scores.append(scores["rouge1"])
        rouge2_scores.append(scores["rouge2"])
        rougeL_scores.append(scores["rougeL"])

    def avg_score(scores):
         # Computes the average precision, recall, and F1-score.
        return {
            "Precision": sum(s.precision for s in scores) / len(scores),
            "Recall": sum(s.recall for s in scores) / len(scores),
            "F1-score": sum(s.fmeasure for s in scores) / len(scores),
        }

    return {
        "ROUGE-1": avg_score(rouge1_scores),
        "ROUGE-2": avg_score(rouge2_scores),
        "ROUGE-L": avg_score(rougeL_scores),
    }

def plot_rouge_scores(results):
     # Plots ROUGE scores as a bar chart.
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    categories = ["Precision", "Recall", "F1-score"]

    # Extracting values for plotting
    scores = {metric: [results[metric][cat] for cat in categories] for metric in metrics}

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.2
    x = range(len(categories))

    for i, metric in enumerate(metrics):
        ax.bar([p + bar_width * i for p in x], scores[metric], bar_width, label=metric)

    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)  # ROUGE scores are between 0 and 1
    ax.set_title("ROUGE Score Evaluation")
    ax.legend()

    plt.show()

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

    # Plot results
    plot_rouge_scores(results)
    
    """

import json
import os
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

# Paths
GROUND_TRUTH_PATH = "results/summaries/ground_truth.json"
GENERATED_SUMMARY_FOLDER = "results/summaries/generated/"  # Folder containing multiple generated summaries


def load_ground_truth():
    """Loads ground truth summaries from the JSON file."""
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["output"] for item in data]  # Extract only the summary texts


def load_generated_summaries():
    """Loads all generated summaries from the folder."""
    summaries = {}

    for filename in os.listdir(GENERATED_SUMMARY_FOLDER):
        if filename.endswith(".txt"):  # Process only text files
            with open(os.path.join(GENERATED_SUMMARY_FOLDER, filename), "r", encoding="utf-8") as f:
                summaries[filename] = f.read().strip().split("\n")  # Split into summary points

    return summaries  # Dictionary {filename: [summary lines]}


def evaluate_summaries(ground_truths, generated_summaries):
    """Calculates ROUGE Precision, Recall, and F1-score."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []

    for filename, gen_summaries in generated_summaries.items():
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

        min_len = min(len(ground_truths), len(gen_summaries))
        gt_trimmed = ground_truths[:min_len]
        gen_trimmed = gen_summaries[:min_len]

        for gt, gen in zip(gt_trimmed, gen_trimmed):
            scores = scorer.score(gt, gen)
            rouge1_scores.append(scores["rouge1"])
            rouge2_scores.append(scores["rouge2"])
            rougeL_scores.append(scores["rougeL"])

        def avg_score(scores):
            """Computes the average precision, recall, and F1-score."""
            return {
                "Precision": sum(s.precision for s in scores) / len(scores),
                "Recall": sum(s.recall for s in scores) / len(scores),
                "F1-score": sum(s.fmeasure for s in scores) / len(scores),
            }

        results.append({
            "filename": filename,
            "ROUGE-1": avg_score(rouge1_scores),
            "ROUGE-2": avg_score(rouge2_scores),
            "ROUGE-L": avg_score(rougeL_scores),
        })

    return results


def compute_average_results(results):
    """Computes the average ROUGE scores across all files."""
    avg_results = {"ROUGE-1": {}, "ROUGE-2": {}, "ROUGE-L": {}}
    categories = ["Precision", "Recall", "F1-score"]

    for metric in avg_results.keys():
        for cat in categories:
            avg_results[metric][cat] = sum(result[metric][cat] for result in results) / len(results)

    return avg_results


def plot_rouge_scores(results, title="ROUGE Score Evaluation"):
    """Plots average ROUGE scores as a bar chart."""
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    categories = ["Precision", "Recall", "F1-score"]

    scores = {metric: [results[metric][cat] for cat in categories] for metric in metrics}

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.2
    x = range(len(categories))

    for i, metric in enumerate(metrics):
        ax.bar([p + bar_width * i for p in x], scores[metric], bar_width, label=metric)

    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)  # ROUGE scores range from 0 to 1
    ax.set_title(title)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # Load data
    ground_truth_summaries = load_ground_truth()
    generated_summaries = load_generated_summaries()

    # Evaluate summaries
    results = evaluate_summaries(ground_truth_summaries, generated_summaries)

    # Print individual results
    print("\n🔹 **Evaluation Results for Each Summary File:**")
    for res in results:
        print(f"\n📂 {res['filename']}:")
        for metric, values in res.items():
            if metric != "filename":
                print(f"{metric}:")
                print(f"   Precision: {values['Precision']:.4f}")
                print(f"   Recall: {values['Recall']:.4f}")
                print(f"   F1-score: {values['F1-score']:.4f}")

    # Compute overall average ROUGE scores
    avg_results = compute_average_results(results)

    # Print average results
    print("\n🔹 **Average ROUGE Scores Across All Files:**")
    for metric, values in avg_results.items():
        print(f"\n{metric}:")
        print(f"   Precision: {values['Precision']:.4f}")
        print(f"   Recall: {values['Recall']:.4f}")
        print(f"   F1-score: {values['F1-score']:.4f}")

    # Plot average results
    plot_rouge_scores(avg_results, title="Average ROUGE Score Across All Files")

