#!/usr/bin/env bash
set -u

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 MODEL_NAME MODEL_PATH GPU_ID" >&2
    exit 2
fi

model_name=$1
model_path=$2
gpu_id=$3
project_dir=$(cd "$(dirname "$0")/.." && pwd)
result_root="$project_dir/result/$model_name"
accelerate_bin=/home/boe/miniconda3/envs/vSLAM/bin/accelerate
status_file="$result_root/status.tsv"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export STREAM3R_DATA_ROOT="${STREAM3R_DATA_ROOT:-/media/boe/HDD_1/data}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg

run_step() {
    local step=$1
    shift
    local safe_step=${step//\//__}
    local log_file="$result_root/logs/$safe_step.log"
    local start_time end_time rc status
    start_time=$(date '+%F %T')
    echo "[$start_time] RETRY $step" > "$log_file"
    "$@" >> "$log_file" 2>&1
    rc=$?
    end_time=$(date '+%F %T')
    if [ "$rc" -eq 0 ]; then status=OK; else status=FAILED; fi
    echo "[$end_time] $status $step (exit=$rc)" | tee -a "$log_file"
    printf '%s\t%s\t%s\t%s\t%s\n' "$step" "$status" "$start_time" "$end_time" "$rc" >> "$status_file"
}

cd "$project_dir" || exit 1
for mode in window full; do
    for dataset in tum sintel scannet; do
        out="$result_root/$mode/relpose/$dataset"
        mkdir -p "$out"
        run_step "$mode/relpose/$dataset" \
            "$accelerate_bin" launch --num_processes 1 \
            --main_process_port "$((29558 + gpu_id))" \
            eval/relpose/launch.py --output_dir "$out" \
            --eval_dataset "$dataset" --model_path "$model_path" --mode "$mode"
    done
done

