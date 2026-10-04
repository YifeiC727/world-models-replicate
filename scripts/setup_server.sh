#!/usr/bin/env bash
# One-time environment setup on the MLDS server (mlds-deepdish1).
# Usage, from the repository root:  bash scripts/setup_server.sh
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/wm}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11 swig   # swig builds Box2D
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch --index-url "$TORCH_INDEX"
"$PY" -m pip install -e '.[dev,carracing]'

"$PY" - <<'EOF'
import torch, gymnasium
print("python ok; torch", torch.__version__, "gymnasium", gymnasium.__version__)
print("cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("compiled archs:", torch.cuda.get_arch_list())
    print("device 0:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn(256, 256, device="cuda"); print("cuda matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
env = gymnasium.make("CarRacing-v3"); env.reset(seed=0); env.close(); print("CarRacing-v3 ok")
EOF
"$PY" -m pytest -q
echo "Activate with:  conda activate $ENV_PREFIX"
