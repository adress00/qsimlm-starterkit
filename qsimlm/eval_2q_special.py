from __future__ import annotations
import argparse
import torch

from .data import make_dataset
from .models import MLPBaseline, Seq2SeqAutoreg
from .metrics import state_fidelity_from_realimag

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "autoreg"], default="autoreg")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    Xte, Yte = make_dataset(args.n_test, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xte = torch.tensor(Xte, device=device)
    Yte = torch.tensor(Yte, device=device)

    if args.model == "mlp":
        model = MLPBaseline().to(device)
    else:
        model = Seq2SeqAutoreg(out_len=8).to(device)

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd)

    if args.model == "mlp":
        yhat = model(Xte)
    else:
        yhat = model(Xte, y_teacher=Yte)

    fid = state_fidelity_from_realimag(Yte, yhat).mean().item()
    print(f"mean_test_fidelity={fid:.4f}")

if __name__ == "__main__":
    main()
