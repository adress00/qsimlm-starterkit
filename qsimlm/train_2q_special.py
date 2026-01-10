from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .data import make_dataset
from .models import MLPBaseline, Seq2SeqAutoreg
from .metrics import state_fidelity_from_realimag

def run(model_name: str, n_train: int, n_test: int, epochs: int, lr: float, seed: int, batch_size: int):
    Xtr, Ytr = make_dataset(n_train, seed=seed)
    Xte, Yte = make_dataset(n_test, seed=seed + 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xtr, device=device)
    Ytr = torch.tensor(Ytr, device=device)
    Xte = torch.tensor(Xte, device=device)
    Yte = torch.tensor(Yte, device=device)

    ds = TensorDataset(Xtr, Ytr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if model_name == "mlp":
        model = MLPBaseline().to(device)
        def forward(x, y):  # noqa
            return model(x)
    elif model_name == "autoreg":
        model = Seq2SeqAutoreg(out_len=8).to(device)
        def forward(x, y):  # noqa
            return model(x, y_teacher=y)
    else:
        raise ValueError("model must be one of: mlp, autoreg")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"{model_name} epoch {ep}/{epochs}")
        for x, y in pbar:
            opt.zero_grad(set_to_none=True)
            yhat = forward(x, y)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))

        model.eval()
        with torch.no_grad():
            if model_name == "mlp":
                yhat = model(Xte)
            else:
                # teacher forcing eval for stability
                yhat = model(Xte, y_teacher=Yte)
            fid = state_fidelity_from_realimag(Yte, yhat).mean().item()
        print(f"[{model_name}] epoch={ep} test_fidelity={fid:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "autoreg"], default="autoreg")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    run(args.model, args.n_train, args.n_test, args.epochs, args.lr, args.seed, args.batch_size)

if __name__ == "__main__":
    main()
