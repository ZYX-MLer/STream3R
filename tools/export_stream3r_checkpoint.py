#!/usr/bin/env python3
"""Export a Lightning/DeepSpeed STream3R checkpoint for HF-style inference."""

import argparse
import json
from pathlib import Path

import torch

from stream3r.models.stream3r import STream3R


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    model_state_path = args.checkpoint / "checkpoint" / "mp_rank_00_model_states.pt"
    if not model_state_path.is_file():
        raise FileNotFoundError(model_state_path)

    checkpoint = torch.load(
        model_state_path, map_location="cpu", mmap=True, weights_only=False
    )
    module_state = checkpoint["module"]
    state_dict = {
        name.removeprefix("net."): tensor
        for name, tensor in module_state.items()
        if name.startswith("net.")
    }

    model = STream3R(img_size=518, patch_size=14, embed_dim=1024)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    provenance = {
        "source_checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "parameter_tensors": len(state_dict),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
