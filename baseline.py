"""Classical models: TF-IDF + One-vs-Rest Logistic Regression.

  tfidf_word       : word 1-2 grams                       (BASELINE)
  tfidf_word_char  : word 1-2 grams + char_wb 2-5 grams   (IMPROVED classical)

Usage:
  python baseline.py --train_langs eng hin deu
  python baseline.py --train_langs eng --test_langs hin --tag zs_eng2hin
"""
import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline, make_union

from common import LABELS, prepare, save_run, tune_thresholds


def make_model(kind):
    # token pattern keeps single characters and emojis/punctuation as tokens,
    # which also works for scripts that sklearn's default pattern mishandles
    word = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                           ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=300_000)
    feats = word if kind == "word" else make_union(word, char)
    clf = OneVsRestClassifier(
        LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced",
                           solver="liblinear"), n_jobs=-1)
    return make_pipeline(feats, clf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_langs", nargs="+", required=True)
    ap.add_argument("--test_langs", nargs="+", default=None)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--tag", default="multi")
    args = ap.parse_args()

    train, val, test = prepare(args.train_langs, args.test_langs,
                               args.data_dir, out_dir=args.out_dir)
    ytr, yva = train[LABELS].values, val[LABELS].values

    for kind, name in [("word", "tfidf_word"), ("word_char", "tfidf_word_char")]:
        t0 = time.time()
        model = make_model(kind).fit(train["clean"], ytr)
        pv = model.predict_proba(val["clean"])
        pt = model.predict_proba(test["clean"])
        thr = tune_thresholds(yva, pv)
        save_run(f"{args.tag}_{name}", test, pt, thr, args.out_dir,
                 extra={"train_langs": args.train_langs,
                        "test_langs": args.test_langs or args.train_langs,
                        "train_seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
