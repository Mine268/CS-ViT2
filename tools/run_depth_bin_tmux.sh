#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-depthbin}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/depth_bin_full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/qnap/data/datasets/webdatasets/depth-bins}"
NUM_FRAMES="${NUM_FRAMES:-1}"
STRIDE="${STRIDE:-1}"
BIN_EDGES="${BIN_EDGES:-0 500 700 900 1100 1000000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$REPO_ROOT/.venv/bin/activate}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux 未安装，请先安装 tmux。" >&2
  exit 1
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "未找到虚拟环境激活脚本: $VENV_ACTIVATE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' 已存在，请先 attach 或更换 session 名称。" >&2
  exit 1
fi

make_cmd() {
  local dataset_name="$1"
  local source_glob="$2"
  local split="$3"
  local log_name="$4"

  cat <<CMD
cd "$REPO_ROOT" && \
source "$VENV_ACTIVATE" && \
mkdir -p "$LOG_DIR" && \
echo "[START] $dataset_name $(date '+%F %T')" | tee -a "$LOG_DIR/$log_name" && \
env PYTHONPATH=. "$PYTHON_BIN" preprocess/depth_bin_wds.py \
  --dataset-name "$dataset_name" \
  --split "$split" \
  --source "$source_glob" \
  --output-root "$OUTPUT_ROOT" \
  --num-frames "$NUM_FRAMES" \
  --stride "$STRIDE" \
  --bin-edges $BIN_EDGES 2>&1 | tee -a "$LOG_DIR/$log_name" ; \
status=\$?; \
echo "[END] $dataset_name status=\$status $(date '+%F %T')" | tee -a "$LOG_DIR/$log_name" ; \
exit \$status
CMD
}

make_serial_cmd() {
  local wait_key="$1"
  local signal_key="$2"
  local title="$3"
  local actual_cmd="$4"

  if [[ -n "$wait_key" ]]; then
    cat <<CMD
echo "[WAIT] waiting for $wait_key"; \
tmux wait-for "$wait_key"; \
$actual_cmd ; \
cmd_status=\$?; \
tmux wait-for -S "$signal_key"; \
exit \$cmd_status
CMD
  else
    cat <<CMD
$actual_cmd ; \
cmd_status=\$?; \
tmux wait-for -S "$signal_key"; \
exit \$cmd_status
CMD
  fi
}

INTERHAND_BASE_CMD="$(make_cmd "InterHand2.6M" "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*" "train" "interhand2.6m_train.log")"
DEXYCB_BASE_CMD="$(make_cmd "DexYCB_s1" "/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/train/*" "train" "dexycb_s1_train.log")"
HO3D_BASE_CMD="$(make_cmd "HO3D_v3" "/mnt/qnap/data/datasets/webdatasets/HO3D_v3/train/*" "train" "ho3d_v3_train.log")"
HOT3D_BASE_CMD="$(make_cmd "HOT3D" "/mnt/qnap/data/datasets/webdatasets/HOT3D/train/*" "train" "hot3d_train.log")"

INTERHAND_CMD="$(make_serial_cmd "" "${SESSION_NAME}_interhand_done" "InterHand2.6M" "$INTERHAND_BASE_CMD")"
DEXYCB_CMD="$(make_serial_cmd "${SESSION_NAME}_interhand_done" "${SESSION_NAME}_dexycb_done" "DexYCB_s1" "$DEXYCB_BASE_CMD")"
HO3D_CMD="$(make_serial_cmd "${SESSION_NAME}_dexycb_done" "${SESSION_NAME}_ho3d_done" "HO3D_v3" "$HO3D_BASE_CMD")"
HOT3D_CMD="$(make_serial_cmd "${SESSION_NAME}_ho3d_done" "${SESSION_NAME}_hot3d_done" "HOT3D" "$HOT3D_BASE_CMD")"

tmux new-session -d -s "$SESSION_NAME" -n interhand "bash -lc '$INTERHAND_CMD'"
tmux new-window -t "$SESSION_NAME" -n dexycb "bash -lc '$DEXYCB_CMD'"
tmux new-window -t "$SESSION_NAME" -n ho3d "bash -lc '$HO3D_CMD'"
tmux new-window -t "$SESSION_NAME" -n hot3d "bash -lc '$HOT3D_CMD'"

echo "tmux session 已创建: $SESSION_NAME"
echo "模式: 4 个窗口串行执行（InterHand2.6M -> DexYCB_s1 -> HO3D_v3 -> HOT3D）"
echo "attach 命令: tmux attach -t $SESSION_NAME"
echo "日志目录: $LOG_DIR"
echo "窗口列表:"
tmux list-windows -t "$SESSION_NAME"
