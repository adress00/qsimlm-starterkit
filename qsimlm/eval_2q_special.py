from __future__ import annotations
import argparse
import torch
import numpy as np

# 确保导入了新定义的 LSTMNonAutoreg
from .data import make_dataset
from .models import MLPBaseline, Seq2SeqAutoreg, LSTMNonAutoreg
from .metrics import state_fidelity_from_realimag


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    # 添加 'lstm' 到选项中
    ap.add_argument("--model", choices=["mlp", "lstm", "autoreg"], default="autoreg")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint .pt file")
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # 1. 准备数据
    Xte, Yte = make_dataset(args.n_test, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 转换为 Tensor
    Xte = torch.tensor(Xte, device=device)
    Yte = torch.tensor(Yte, device=device)

    # 2. 初始化模型 (结构参数必须与训练时一致)
    print(f"Loading {args.model} model...")
    if args.model == "mlp":
        model = MLPBaseline().to(device)
    elif args.model == "lstm":
        # 新增 LSTM 支持
        model = LSTMNonAutoreg(out_len=8).to(device)
    else:
        # Autoreg 模型
        model = Seq2SeqAutoreg(out_len=8).to(device)

    # 3. 加载权重
    print(f"Loading weights from {args.ckpt}...")
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # 4. 推理 (核心修改点)
    # 之前你的代码是: model(Xte, y_teacher=Yte) -> 这是作弊(Teacher Forcing)
    # 修改后: 直接调用 model(Xte)
    # - 对于 MLP/LSTM: 进行普通前向传播
    # - 对于 Autoreg: 会触发 models.py 里的 'autoregressive inference' (Greedy Search)
    print("Running inference...")
    yhat = model(Xte)

    # 5. 计算指标
    # 获取每个样本的 fidelity (形状: [n_test])
    fids = state_fidelity_from_realimag(Yte, yhat)

    # 转换为 numpy 方便统计
    fids_np = fids.cpu().numpy()

    print("-" * 30)
    print(f"Model: {args.model}")
    print(f"Mean Fidelity:   {np.mean(fids_np):.4f}")
    print(f"Median Fidelity: {np.median(fids_np):.4f}")
    print(f"Min (Worst) Fid: {np.min(fids_np):.4f}")  # 关注这个指标，Autoreg 通常这里更高
    print(f"Max (Best) Fid:  {np.max(fids_np):.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()