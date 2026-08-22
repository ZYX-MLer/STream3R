#!/bin/bash
set -e

workdir='.'
export STREAM3R_DATA_ROOT="${STREAM3R_DATA_ROOT:-/media/boe/HDD_1/data}"

model_name='stream3r'

output_dir="${workdir}/eval_results/mv_recon/${model_name}/"
echo "$output_dir"

python eval/mv_recon/launch.py \
    --output_dir="$output_dir" \
    --size=518 \
    --model_name="stream3r" \
