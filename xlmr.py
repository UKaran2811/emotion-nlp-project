"""Fine-tune XLM-RoBERTa for multi-label emotion detection (plain PyTorch).

Usage (Colab GPU recommended):
  python xlmr.py --train_langs eng hin deu --epochs 4
  python xlmr.py --train_langs eng --test_langs hin --tag zs_eng2hin
  python xlmr.py --model cardiffnlp/twitter-xlm-roberta-base --tag multi_twxlmr
"""
import argparse
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import LABELS, evaluate, prepare, save_run, tune_thresholds


class TextDS(Dataset):
    def __init__(self, df):
        self.t = df["clean"].tolist()
        self.y = df[LABELS].values.astype("float32")

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], self.y[i]


def make_collate(tok, max_len):
    def collate(batch):
        texts, ys = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc = dict(enc)
        enc["labels"] = torch.tensor(np.stack(ys))
        return enc
    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(torch.sigmoid(model(**batch).logits).float().cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_langs", nargs="+", required=True)
    ap.add_argument("--test_langs", nargs="+", default=None)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pos_weight", action="store_true")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--tag", default="multi")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print("device:", device)

    train, val, test = prepare(args.train_langs, args.test_langs,
                               args.data_dir, out_dir=args.out_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    collate = make_collate(tok, args.max_len)
    train_dl = DataLoader(TextDS(train), batch_size=args.bs, shuffle=True,
                          collate_fn=collate)
    val_dl = DataLoader(TextDS(val), batch_size=64, collate_fn=collate)
    test_dl = DataLoader(TextDS(test), batch_size=64, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        problem_type="multi_label_classification").to(device)

    # positive-class weight (sqrt-damped) to counter label imbalance
    pos = train[LABELS].values.sum(0).clip(min=1)
    neg = len(train) - pos
    pw = np.ones(len(LABELS)) if args.no_pos_weight else np.sqrt(neg / pos)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    yva = val[LABELS].values
    best_f1, best_val_p, best_test_p, best_ep, history = -1, None, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_dl:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item() * len(labels); n += len(labels)

        pv, pt = predict(model, val_dl, device), predict(model, test_dl, device)
        vf1 = evaluate(yva, (pv >= 0.5).astype(int))["macro_f1"]
        history.append({"epoch": ep, "train_loss": run_loss / n, "val_macro_f1": vf1})
        print(f"epoch {ep}: loss={run_loss / n:.4f} val macro-F1@0.5={vf1:.4f}")
        if vf1 > best_f1:  # selection uses VALIDATION only
            best_f1, best_val_p, best_test_p, best_ep = vf1, pv, pt, ep

    thr = tune_thresholds(yva, best_val_p)
    at_half = evaluate(test[LABELS].values, (best_test_p >= 0.5).astype(int))
    save_run(f"{args.tag}_xlmr", test, best_test_p, thr, args.out_dir,
             extra={"model_name": args.model, "best_epoch": best_ep,
                    "history": history, "overall_at_0.5": at_half,
                    "train_langs": args.train_langs,
                    "test_langs": args.test_langs or args.train_langs,
                    "train_seconds": round(time.time() - t0, 1),
                    "seed": args.seed})


if __name__ == "__main__":
    main()
