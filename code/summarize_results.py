"""Create report tables and the required loss-curve figure from runner output."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    args = p.parse_args()
    root = Path(args.results)
    data = {(model, seed): json.loads((root / f"{model}_seed{seed}.json").read_text())
            for model in "12345" for seed in (2026, 577)}
    keys = ["overall", "non_punctuation", "unseen", "ambiguous"]
    rows = []
    for model in "12345":
        a, b = data[model, 2026], data[model, 577]
        row = {"model": model, "trainable_parameters": a["trainable_parameters"],
               "best_epoch_seed2026": a["best_epoch"], "best_epoch_seed577": b["best_epoch"]}
        for key in keys:
            if a["metrics"][key]["n"] != b["metrics"][key]["n"]:
                raise ValueError("Subset counts differ across seeds")
            values = [r["metrics"][key]["accuracy"] for r in (a, b)]
            row[key + "_n"] = a["metrics"][key]["n"]
            row[key + "_mean_percent"] = float(np.mean(values) * 100) if values[0] is not None else "N/A"
            row[key + "_std_pp"] = float(np.std(values, ddof=0) * 100) if values[0] is not None else "N/A"
        rows.append(row)
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "subword_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pieces", "n", "mean_percent", "std_pp"])
        for key in ["pieces_1", "pieces_2", "pieces_3plus"]:
            metrics = [data["4", s]["metrics"][key] for s in (2026, 577)]
            values = [m["accuracy"] for m in metrics]
            writer.writerow([key, metrics[0]["n"],
                float(np.mean(values)*100) if values[0] is not None else "N/A",
                float(np.std(values, ddof=0)*100) if values[0] is not None else "N/A"])
    (root / "selected_errors.json").write_text(json.dumps(data["2",2026]["errors"], indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(7, 9), sharex=True)
    for ax, model in zip(axes, "12345"):
        history = data[model, 2026]["history"]
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Training")
        ax.plot([r["epoch"] for r in history], [r["valid_loss"] for r in history], label="Validation")
        ax.set_title(f"Configuration {model}", fontsize=10)
        ax.set_ylabel("CE loss")
        ax.grid(alpha=.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(root / "learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote summary.csv, subword_counts.csv, selected_errors.json, learning_curves.pdf to {root}")


if __name__ == "__main__":
    main()
