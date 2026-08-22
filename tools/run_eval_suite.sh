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
python_bin=/home/boe/miniconda3/envs/vSLAM/bin/python
accelerate_bin=/home/boe/miniconda3/envs/vSLAM/bin/accelerate

export CUDA_VISIBLE_DEVICES="$gpu_id"
export STREAM3R_DATA_ROOT="${STREAM3R_DATA_ROOT:-/media/boe/HDD_1/data}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg

mkdir -p "$result_root/logs"
status_file="$result_root/status.tsv"
printf 'step\tstatus\tstart_time\tend_time\texit_code\n' > "$status_file"

run_step() {
    local step=$1
    shift
    local safe_step=${step//\//__}
    local log_file="$result_root/logs/$safe_step.log"
    local start_time end_time rc status
    start_time=$(date '+%F %T')
    echo "[$start_time] START $step" | tee "$log_file"
    printf 'COMMAND:' >> "$log_file"
    printf ' %q' "$@" >> "$log_file"
    printf '\n' >> "$log_file"
    "$@" >> "$log_file" 2>&1
    rc=$?
    end_time=$(date '+%F %T')
    if [ "$rc" -eq 0 ]; then status=OK; else status=FAILED; fi
    echo "[$end_time] $status $step (exit=$rc)" | tee -a "$log_file"
    printf '%s\t%s\t%s\t%s\t%s\n' "$step" "$status" "$start_time" "$end_time" "$rc" >> "$status_file"
    return 0
}

cd "$project_dir" || exit 1

for mode in window full; do
    for dataset in sintel bonn kitti nyu; do
        out="$result_root/$mode/monodepth/$dataset"
        mkdir -p "$out"
        run_step "$mode/monodepth/$dataset/inference" \
            "$python_bin" eval/monodepth/launch.py \
            --output_dir "$out" --eval_dataset "$dataset" \
            --model_path "$model_path" --mode "$mode"
        run_step "$mode/monodepth/$dataset/metrics" \
            "$python_bin" eval/monodepth/eval_metrics.py \
            --output_dir "$out" --eval_dataset "$dataset"
    done

    for dataset in sintel bonn kitti; do
        out="$result_root/$mode/video_depth/$dataset"
        mkdir -p "$out"
        run_step "$mode/video_depth/$dataset/inference" \
            "$python_bin" eval/video_depth/launch.py \
            --output_dir "$out" --eval_dataset "$dataset" \
            --model_path "$model_path" --mode "$mode"
        run_step "$mode/video_depth/$dataset/metrics" \
            "$python_bin" eval/video_depth/eval_depth.py \
            --output_dir "$out" --eval_dataset "$dataset" --align scale
    done

    for dataset in tum sintel scannet; do
        out="$result_root/$mode/relpose/$dataset"
        mkdir -p "$out"
        run_step "$mode/relpose/$dataset" \
            "$accelerate_bin" launch --num_processes 1 --main_process_port "$((29558 + gpu_id))" \
            eval/relpose/launch.py --output_dir "$out" \
            --eval_dataset "$dataset" --model_path "$model_path" --mode "$mode"
    done

    out="$result_root/$mode/mv_recon"
    mkdir -p "$out"
    run_step "$mode/mv_recon" \
        "$python_bin" eval/mv_recon/launch.py \
        --output_dir "$out" --size 518 --model_name stream3r \
        --model_path "$model_path" --mode "$mode"
done

date '+%F %T' > "$result_root/SUITE_FINISHED"
