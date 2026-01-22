#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -d ".venv" ]; then
  echo "[run.sh] Creating venv..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[run.sh] Upgrading pip..."
python -m pip install -U pip

echo "[run.sh] Installing requirements..."
pip install -r requirements.txt

echo "[run.sh] Running a short training (autoreg)..."
# 你可以尝试修改参数，例如：
# --model lstm        : 运行 LSTM 模型
# --noisy             : 开启噪声模拟
# --use_trig          : 使用三角函数特征
python -m qsimlm.train_2q_special --model autoreg --n_train 20000 --n_test 2000 --epochs 8

echo "[run.sh] Done."
