"""Shared utilities: data loading, preprocessing, metrics, result saving.

Dataset: BRIGHTER / SemEval-2025 Task 11 (Track A, multi-label emotions).
Expected CSV layout (also used as cache): data/<lang>_<split>.csv
with columns: text, anger, fear, joy, sadness, surprise (0/1).
You can drop in your own CSVs with the same layout to use any other dataset.
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, hamming_loss, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

HF_ID = "brighter-dataset/BRIGHTER-emotion-categories"
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
SEED = 42

# ----------------------------------------------------------------------------
# Preprocessing (language-agnostic: no stopword removal, no stemming)
# ----------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")
WS_RE = re.compile(r"\s+")


def clean_text(text):
    t = unicodedata.normalize("NFKC", str(text))
    t = URL_RE.sub(" <url> ", t)
    t = USER_RE.sub(" <user> ", t)
    t = HASH_RE.sub(r" \1 ", t)          # keep the hashtag word
    t = REPEAT_RE.sub(r"\1\1", t)        # sooooo -> soo
    t = t.lower()
    return WS_RE.sub(" ", t).strip()


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _load_split(lang, split, data_dir):
    path = Path(data_dir) / f"{lang}_{split}.csv"
    if path.exists():
        return pd.read_csv(path)
    from datasets import load_dataset  # needs internet the first time
    df = load_dataset(HF_ID, lang, split=split).to_pandas()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _normalise(df, lang):
    if "text" not in df.columns:
        raise ValueError(f"[{lang}] no 'text' column; found {list(df.columns)}")
    df = df.dropna(subset=["text"]).copy()
    for lab in LABELS:
        if lab not in df.columns:
            print(f"[warn] {lang}: label '{lab}' missing, filled with 0")
            df[lab] = 0
    df[LABELS] = df[LABELS].fillna(0).astype(int)
    df["lang"] = lang
    return df[["text", "lang"] + LABELS].reset_index(drop=True)


def prepare(train_langs, test_langs=None, data_dir="data", val_frac=0.1,
            out_dir="results"):
    """Return train, val, test DataFrames (with a 'clean' column).

    train/val  <- 'train' split of train_langs (val = stratified-by-language 10%)
    test       <- official 'dev' split of test_langs (held-out; test labels may
                  not be public). test_langs != train_langs gives zero-shot.
    """
    test_langs = test_langs or train_langs
    train = pd.concat([_normalise(_load_split(l, "train", data_dir), l)
                       for l in train_langs], ignore_index=True)
    test = pd.concat([_normalise(_load_split(l, "dev", data_dir), l)
                      for l in test_langs], ignore_index=True)
    train, val = train_test_split(train, test_size=val_frac, random_state=SEED,
                                  stratify=train["lang"])
    train, val, test = train.copy(), val.copy(), test.copy()
    for d in (train, val, test):
        d["clean"] = d["text"].map(clean_text)
    train, val, test = [d.reset_index(drop=True) for d in (train, val, test)]
    dataset_stats(train, val, test, out_dir)
    return train, val, test


def dataset_stats(train, val, test, out_dir="results"):
    rows = []
    for name, d in [("train", train), ("val", val), ("test", test)]:
        for lang, g in d.groupby("lang"):
            n_lab = g[LABELS].sum(axis=1)
            row = {"split": name, "lang": lang, "n": len(g),
                   "avg_words": round(g["text"].str.split().str.len().mean(), 1),
                   "avg_chars": round(g["text"].str.len().mean(), 1)}
            row.update({lab: int(g[lab].sum()) for lab in LABELS})
            row["no_emotion_%"] = round(100 * (n_lab == 0).mean(), 1)
            row["multi_label_%"] = round(100 * (n_lab > 1).mean(), 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "dataset_stats.csv", index=False)
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    out = {"subset_accuracy": float(accuracy_score(y_true, y_pred)),
           "hamming_loss": float(hamming_loss(y_true, y_pred)),
           "samples_f1": float(f1_score(y_true, y_pred, average="samples",
                                        zero_division=0))}
    for avg in ("micro", "macro"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        out.update({f"{avg}_precision": float(p), f"{avg}_recall": float(r),
                    f"{avg}_f1": float(f)})
    return out


def per_label(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    return {lab: {"precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(LABELS)}


def per_lang(df, y_pred):
    out = {}
    y_true = df[LABELS].values
    for lang in sorted(df["lang"].unique()):
        m = (df["lang"] == lang).values
        out[lang] = {
            "n": int(m.sum()),
            "macro_f1": float(f1_score(y_true[m], y_pred[m], average="macro",
                                       zero_division=0)),
            "micro_f1": float(f1_score(y_true[m], y_pred[m], average="micro",
                                       zero_division=0))}
    return out


def tune_thresholds(y_true, probs, grid=None):
    """Pick the F1-maximising threshold per emotion on VALIDATION data."""
    grid = np.arange(0.1, 0.91, 0.05) if grid is None else grid
    thr = []
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], (probs[:, j] >= t).astype(int),
                           zero_division=0) for t in grid]
        thr.append(float(grid[int(np.argmax(scores))]))
    return np.array(thr)


def save_run(name, test_df, probs, thr, out_dir="results", extra=None):
    """Save metrics (JSON) and per-example predictions (CSV) for one model."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    y_true = test_df[LABELS].values
    y_pred = (probs >= np.asarray(thr)).astype(int)
    res = {"model": name,
           "overall": evaluate(y_true, y_pred),
           "per_label": per_label(y_true, y_pred),
           "per_lang": per_lang(test_df, y_pred),
           "thresholds": [float(t) for t in np.broadcast_to(thr, len(LABELS))]}
    if extra:
        res.update(extra)
    with open(Path(out_dir) / f"{name}.json", "w") as f:
        json.dump(res, f, indent=2)
    pred = test_df[["text", "lang"]].copy()
    for i, lab in enumerate(LABELS):
        pred[f"true_{lab}"] = y_true[:, i]
        pred[f"pred_{lab}"] = y_pred[:, i]
        pred[f"prob_{lab}"] = probs[:, i].round(4)
    pred.to_csv(Path(out_dir) / f"{name}_preds.csv", index=False)
    o = res["overall"]
    print(f"[{name}] macro-F1={o['macro_f1']:.4f} micro-F1={o['micro_f1']:.4f} "
          f"subset-acc={o['subset_accuracy']:.4f} hamming={o['hamming_loss']:.4f}")
    return res


if __name__ == "__main__":
    # Prints the language configs available on Hugging Face.
    from datasets import get_dataset_config_names
    print(get_dataset_config_names(HF_ID))
