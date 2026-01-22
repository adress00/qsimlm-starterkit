from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# 确保你的文件夹里有这些文件
from .data import make_dataset
# 引入所有三个模型类
from .models import MLPBaseline, Seq2SeqAutoreg, LSTMNonAutoreg
from .models import MLPBaseline, Seq2SeqAutoreg, LSTMNonAutoreg
from .metrics import state_fidelity_from_realimag, density_matrix_fidelity


def run(model_name: str, n_train: int, n_test: int, epochs: int, lr: float, seed: int, batch_size: int,
        noisy: bool, noise_prob: float, use_trig: bool):
    # 1. 准备数据
    Xtr, Ytr = make_dataset(n_train, seed=seed, noisy=noisy, noise_prob=noise_prob)
    Xte, Yte = make_dataset(n_test, seed=seed + 1, noisy=noisy, noise_prob=noise_prob)

    # 特征工程: (N, 4, 3) -> (N, 4, 6) if use_trig
    if use_trig:
        print("--- Feature Engineering: Using Sin/Cos features ---")
        # X: (N, 4, 3) radians
        # New X: cat([sin(X), cos(X)], dim=-1)
        Xtr_sin, Xtr_cos = np.sin(Xtr), np.cos(Xtr)
        Xtr = np.concatenate([Xtr_sin, Xtr_cos], axis=-1) # (N, 4, 6)

        Xte_sin, Xte_cos = np.sin(Xte), np.cos(Xte)
        Xte = np.concatenate([Xte_sin, Xte_cos], axis=-1)

        in_feat = 6
        mlp_in_dim = 4 * 6
    else:
        in_feat = 3
        mlp_in_dim = 4 * 3

    out_dim = 32 if noisy else 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xtr, device=device)
    Ytr = torch.tensor(Ytr, device=device)
    Xte = torch.tensor(Xte, device=device)
    Yte = torch.tensor(Yte, device=device)

    ds = TensorDataset(Xtr, Ytr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型与定义训练行为
    print(f"--- Initializing {model_name} on {device} (in_feat={in_feat}) ---")

    if model_name == "mlp":
        model = MLPBaseline(in_dim=mlp_in_dim, out_dim=out_dim).to(device)

        # MLP 不需要 teacher，忽略 y
        def train_forward(x, y):
            return model(x)

    elif model_name == "lstm":
        # 新增的 LSTM Baseline
        model = LSTMNonAutoreg(in_feat=in_feat, out_len=out_dim).to(device)

        # LSTM 也不需要 teacher，忽略 y
        def train_forward(x, y):
            return model(x)

    elif model_name == "autoreg":
        model = Seq2SeqAutoreg(in_feat=in_feat, out_len=out_dim).to(device)

        # Autoreg 训练时必须开启 Teacher Forcing
        def train_forward(x, y):
            return model(x, y_teacher=y)

    else:
        raise ValueError("model must be one of: mlp, lstm, autoreg")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 3. 训练循环
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"[{model_name}] Epoch {ep}/{epochs}")

        avg_loss = 0.0
        steps = 0

        for x, y in pbar:
            opt.zero_grad(set_to_none=True)

            # 使用上面定义的专用 forward 函数
            yhat = train_forward(x, y)

            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()

            current_loss = loss.item()
            avg_loss += current_loss
            steps += 1
            pbar.set_postfix(loss=current_loss)

        # 4. Evaluation (关键修改)
        model.eval()
        with torch.no_grad():
            # 这里的调用逻辑统一了：
            # - 对于 MLP/LSTM: model(Xte) 直接前向传播
            # - 对于 Autoreg: model(Xte) 等价于 model(Xte, y_teacher=None)
            #   这会自动触发 models.py 里的 'autoregressive inference' 循环
            #   这才是真正的 "Testing"，看模型能不能自己一步步生成正确的态
            yhat_test = model(Xte)

            # 计算 Fidelity
            # 注意：Yte 是真实值，yhat_test 是预测值
            if noisy:
                fid = density_matrix_fidelity(Yte, yhat_test).mean().item()
            else:
                fid = state_fidelity_from_realimag(Yte, yhat_test).mean().item()

        print(f"Epoch {ep} finished | Avg Loss: {avg_loss / steps:.6f} | Test Fidelity: {fid:.4f}")


def main():
    ap = argparse.ArgumentParser()
    # 在 choices 里加入了 'lstm'
    ap.add_argument("--model", choices=["mlp", "lstm", "autoreg"], default="autoreg")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)  # 建议稍微增加 epoch 以观察收敛差异
    ap.add_argument("--lr", type=float, default=1e-3)  # Transformer 推荐 1e-3 或 3e-4
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--noisy", action="store_true", help="Use noisy simulation (density matrix targets)")
    ap.add_argument("--noise_prob", type=float, default=0.01)
    ap.add_argument("--use_trig", action="store_true", help="Use sin/cos input features")
    args = ap.parse_args()

    run(args.model, args.n_train, args.n_test, args.epochs, args.lr, args.seed, args.batch_size, args.noisy,
        args.noise_prob, args.use_trig)


if __name__ == "__main__":
    main()