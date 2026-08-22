#!/bin/bash
set -e

workdir='.'
export STREAM3R_DATA_ROOT="${STREAM3R_DATA_ROOT:-/media/boe/HDD_1/data}"

datasets=('sintel' 'bonn' 'kitti' 'nyu')
model_name='stream3r'

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${model_name}/${data}"
    echo "$output_dir"

    python eval/monodepth/launch.py \
    --output_dir="$output_dir" \
    --eval_dataset="$data" \

    python eval/monodepth/eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done
