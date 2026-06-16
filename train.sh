#!/bin/bash
set -e

# 切换到脚本所在目录，确保相对路径稳定
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# 确保使用正确的conda环境
# 注意：如果脚本在没有conda初始化的shell中运行，可能需要调整这里
source ~/.bashrc
# 兼容非交互shell：conda 命令可能不存在
if ! command -v conda >/dev/null 2>&1; then
	if [ -f "/data1/sunchao/anaconda3/etc/profile.d/conda.sh" ]; then
		source "/data1/sunchao/anaconda3/etc/profile.d/conda.sh"
	elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
		source "$HOME/anaconda3/etc/profile.d/conda.sh"
	fi
fi
# 尝试激活环境，如果失败则打印警告但不退出（因为可能已经在环境中）
conda activate UniSpine_GS || echo "Warning: conda activate failed, assuming environment is already active."

echo "========================================"
echo "开始批量训练 + 评估 data 一级目录所有 pickle"
echo "========================================"

if command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
else
	echo "Error: python/python3 not found in PATH"
	exit 1
fi

"${PYTHON_BIN}" batch_train_eval.py --project_root "${SCRIPT_DIR}" --data_dir "data"

echo "========================================"
echo "全部流程完成！"
echo "汇总结果保存在: output/batch_eval_summary.json"
echo "汇总结果保存在: output/batch_eval_summary.txt"
echo "========================================"
