#!/usr/bin/env bash
set -euo pipefail

# Always run from the repository root, no matter where the script is called from.
cd "$(dirname "$0")"

echo "=== System setup ==="
export DEBIAN_FRONTEND=noninteractive

# ubuntu:22.04 may not include Python, pip, or venv.
if ! command -v python3 >/dev/null 2>&1; then
    apt-get update
    apt-get install -y python3 python3-pip python3-venv
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
    apt-get update
    apt-get install -y python3-venv
fi

echo "=== Creating virtual environment ==="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Installing dependencies ==="
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "=== Running complete training + evaluation pipeline ==="
mkdir -p data outputs logs

# For final submission, keep the default values below aligned with your report.
# To do a faster local smoke test, override them like:
# EPISODES=50 TRIALS=5 HARD_TRIALS=3 bash run.sh
EPISODES="${EPISODES:-5000}"
TRIALS="${TRIALS:-50}"
HARD_TRIALS="${HARD_TRIALS:-30}"

python src/main.py \
  --data-dir data \
  --output-dir outputs \
  --episodes "$EPISODES" \
  --trials "$TRIALS" \
  --hard-trials "$HARD_TRIALS" \
  2>&1 | tee logs/run.log

echo "=== Done ==="
echo "Main outputs:"
echo "  outputs/results.json"
echo "  outputs/evaluation_results.csv"
echo "  outputs/hard_graph_results.csv"
echo "  outputs/actor_critic_model.pt"
echo "  outputs/*.png"
echo "  logs/run.log"