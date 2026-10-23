"""Comparison tables, charts and error analysis.

Usage: python report.py --tag multi
Outputs (in results/): comparison_<tag>.csv/.md, fig_*.png, error_*.csv
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LABELS

METRICS = ["micro_f1", "macro_f1", "macro_precision", "macro_recall",
           "subset_accuracy", "hamming_loss"]


def load_runs(out_dir, tag):
    runs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, f"{tag}_*.json"))):
        with open(p) as f:
            r = json.load(f)
        runs[r["model"].replace(f"{tag}_", "", 1)] = r
    return runs


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def grouped_bar(data, title, ylabel, path):
    """data: {series_name: {category: value}}"""
    cats = sorted({c for d in data.values() for c in d})
    w = 0.8 / max(len(data), 1)
    x = np.arange(len(cats))
    plt.figure(figsize=(8, 4))
    for i, (name, d) in enumerate(data.items()):
        plt.bar(x + i * w, [d.get(c, 0) for c in cats], w, label=name)
    plt.xticks(x + w * (len(data) - 1) / 2, cats)
    plt.ylabel(ylabel); plt.title(title); plt.ylim(0, 1)
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def paired_bootstrap(yt, ya, yb, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        fa = f1_score(yt[idx], ya[idx], average="macro", zero_division=0)
        fb = f1_score(yt[idx], yb[idx], average="macro", zero_division=0)
        diffs.append(fb - fa)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def error_analysis(pred_csv, out_dir, tag, name):
    df = pd.read_csv(pred_csv)
    yt = df[[f"true_{l}" for l in LABELS]].values
    yp = df[[f"pred_{l}" for l in LABELS]].values
    pr = df[[f"prob_{l}" for l in LABELS]].values
    miss, spur = (yt == 1) & (yp == 0), (yt == 0) & (yp == 1)

    # 1) FP / FN per emotion
    fpfn = pd.DataFrame({"emotion": LABELS, "false_negatives": miss.sum(0),
                         "false_positives": spur.sum(0), "support": yt.sum(0)})
    fpfn.to_csv(f"{out_dir}/error_fp_fn_{tag}.csv", index=False)

    # 2) row-level error rate per language and per length bin
    df["row_error"] = (yt != yp).any(axis=1)
    df["hamming_row"] = (yt != yp).mean(axis=1)
    by_lang = df.groupby("lang").agg(n=("row_error", "size"),
                                     any_error_rate=("row_error", "mean"),
                                     hamming=("hamming_row", "mean")).round(3)
    by_lang.to_csv(f"{out_dir}/error_by_language_{tag}.csv")
    df["n_words"] = df["text"].astype(str).str.split().str.len()
    try:
        df["len_bin"] = pd.qcut(df["n_words"], 4, duplicates="drop")
    except ValueError:
        df["len_bin"] = "all"
    by_len = df.groupby("len_bin", observed=True).agg(
        n=("row_error", "size"), any_error_rate=("row_error", "mean")).round(3)
    by_len.to_csv(f"{out_dir}/error_by_length_{tag}.csv")

    # 3) confusion pairs: true-but-missed (rows) vs wrongly-predicted (cols)
    conf = miss.astype(int).T @ spur.astype(int)
    pd.DataFrame(conf, index=LABELS, columns=LABELS).to_csv(
        f"{out_dir}/error_confusion_pairs_{tag}.csv")
    plt.figure(figsize=(5, 4))
    plt.imshow(conf, cmap="Blues")
    plt.xticks(range(len(LABELS)), LABELS, rotation=45)
    plt.yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, conf[i, j], ha="center", va="center", fontsize=8)
    plt.xlabel("wrongly predicted"); plt.ylabel("true but missed")
    plt.title(f"Confusion pairs ({name})"); plt.tight_layout()
    plt.savefig(f"{out_dir}/fig_confusion_pairs_{tag}.png", dpi=200); plt.close()

    # 4) most confident mistakes, for manual categorisation
    df["error_score"] = np.abs(pr - yt).sum(axis=1)
    cols = (["lang", "text"] + [f"true_{l}" for l in LABELS]
            + [f"pred_{l}" for l in LABELS] + ["error_score"])
    df[df["row_error"]].sort_values("error_score", ascending=False).head(100)[
        cols].to_csv(f"{out_dir}/errors_sample_{tag}.csv", index=False)

    print("\nFP/FN per emotion:\n", fpfn.to_string(index=False))
    print("\nError by language:\n", by_lang)
    print("\nError by length:\n", by_len)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="multi")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    out, tag = args.out_dir, args.tag

    runs = load_runs(out, tag)
    if not runs:
        raise SystemExit(f"No results/{tag}_*.json found. Run baseline.py / xlmr.py first.")

    # ---- comparison table
    rows = []
    for name, r in runs.items():
        row = {"model": name}
        row.update({m: round(r["overall"][m], 4) for m in METRICS})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(f"{out}/comparison_{tag}.csv", index=False)
    with open(f"{out}/comparison_{tag}.md", "w") as f:
        f.write(to_markdown(table))
    print(table.to_string(index=False))

    # ---- charts
    grouped_bar({n: {"macro-F1": r["overall"]["macro_f1"],
                     "micro-F1": r["overall"]["micro_f1"]}
                 for n, r in runs.items()},
                "Overall F1 by model", "F1", f"{out}/fig_overall_{tag}.png")
    grouped_bar({n: {l: r["per_lang"][l]["macro_f1"] for l in r["per_lang"]}
                 for n, r in runs.items()},
                "Macro-F1 per language", "macro-F1", f"{out}/fig_per_language_{tag}.png")
    grouped_bar({n: {l: r["per_label"][l]["f1"] for l in LABELS}
                 for n, r in runs.items()},
                "F1 per emotion", "F1", f"{out}/fig_per_emotion_{tag}.png")

    # ---- error analysis on the best model
    best = max(runs, key=lambda n: runs[n]["overall"]["macro_f1"])
    print(f"\nBest model by macro-F1: {best}")
    error_analysis(f"{out}/{tag}_{best}_preds.csv", out, tag, best)

    # ---- paired bootstrap: baseline vs best
    base = "tfidf_word"
    if base in runs and best != base:
        a = pd.read_csv(f"{out}/{tag}_{base}_preds.csv")
        b = pd.read_csv(f"{out}/{tag}_{best}_preds.csv")
        if len(a) == len(b) and (a["text"] == b["text"]).all():
            yt = a[[f"true_{l}" for l in LABELS]].values
            ya = a[[f"pred_{l}" for l in LABELS]].values
            yb = b[[f"pred_{l}" for l in LABELS]].values
            d, lo, hi = paired_bootstrap(yt, ya, yb)
            print(f"\nPaired bootstrap macro-F1 ({best} - {base}): "
                  f"{d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
