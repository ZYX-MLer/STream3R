#!/usr/bin/env python3
"""Collect STream3R evaluation artifacts into per-model and comparison reports."""

import argparse
import csv
import json
import re
from pathlib import Path


MODES = ("window", "full")
TRAINED_MODEL_CODE = "STream3R-local-20260822-R01"
TRAINED_MODEL_ANNOTATION = "首次自主训练结果"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def relpose_metrics(path):
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    matches = re.findall(
        r"Average ATE:\s*([\d.eE+-]+), Average RPE trans:\s*([\d.eE+-]+), "
        r"Average RPE rot:\s*([\d.eE+-]+)",
        text,
    )
    if not matches:
        return {"error": "average metrics not found", "oom_count": text.count("OOM error")}
    ate, trans, rot = matches[-1]
    return {
        "ate": float(ate),
        "rpe_trans": float(trans),
        "rpe_rot": float(rot),
        "oom_count": text.count("OOM error"),
    }


def mv_metrics(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return None
    mean_lines = [line for line in lines if line.lstrip().startswith("mean")]
    if not mean_lines:
        return None
    return {
        key: float(value)
        for key, value in re.findall(r"([A-Za-z0-9_]+):\s*([\d.eE+-]+)", mean_lines[-1])
        if key != "mean"
    }


def status_summary(model_dir):
    path = model_dir / "status.tsv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    latest = {}
    for row in rows:
        latest[row["step"]] = row
    return list(latest.values())


def collect_model(model_dir, model_name):
    report = {
        "model": model_name,
        "model_code": TRAINED_MODEL_CODE if model_name == "trained" else "STream3R-official",
        "annotation": TRAINED_MODEL_ANNOTATION if model_name == "trained" else "官方发布权重",
        "status": status_summary(model_dir),
        "results": {},
    }
    for mode in MODES:
        mode_results = {}
        mono = {}
        for dataset in ("sintel", "bonn", "kitti", "nyu"):
            mono[dataset] = read_json(model_dir / mode / "monodepth" / dataset / "metric.json")
        mode_results["monodepth"] = mono

        video = {}
        for dataset in ("sintel", "bonn", "kitti"):
            video[dataset] = read_json(
                model_dir / mode / "video_depth" / dataset / "result_scale.json"
            )
        mode_results["video_depth"] = video

        pose = {}
        for dataset in ("tum", "sintel", "scannet"):
            pose[dataset] = relpose_metrics(
                model_dir / mode / "relpose" / dataset / "_error_log.txt"
            )
        mode_results["relpose"] = pose

        recon = {}
        for dataset in ("7scenes", "NRGBD"):
            recon[dataset] = mv_metrics(
                model_dir / mode / "mv_recon" / dataset / "logs_all.txt"
            )
        mode_results["mv_recon"] = recon
        report["results"][mode] = mode_results
    return report


def metric_rows(report):
    for mode, tasks in report["results"].items():
        for task, datasets in tasks.items():
            for dataset, metrics in datasets.items():
                if not isinstance(metrics, dict):
                    yield mode, task, dataset, "status", "missing"
                    continue
                for metric, value in metrics.items():
                    yield mode, task, dataset, metric, value


TASK_SPECS = {
    "monodepth": {
        "title": "单帧深度评测（Single-frame Depth）",
        "datasets": ("sintel", "bonn", "kitti", "nyu"),
        "dataset_labels": {
            "sintel": "Sintel",
            "bonn": "Bonn",
            "kitti": "KITTI",
            "nyu": "NYU-v2",
        },
        "metrics": (("Abs Rel", "AbsRel ↓", "min"), ("δ < 1.25", "δ<1.25 ↑", "max")),
    },
    "video_depth": {
        "title": "视频深度评测（Video Depth，逐序列尺度对齐）",
        "datasets": ("sintel", "bonn", "kitti"),
        "dataset_labels": {"sintel": "Sintel", "bonn": "Bonn", "kitti": "KITTI"},
        "metrics": (("Abs Rel", "AbsRel ↓", "min"), ("δ < 1.25", "δ<1.25 ↑", "max")),
    },
    "relpose": {
        "title": "相机位姿评测（Camera Pose）",
        "datasets": ("sintel", "tum", "scannet"),
        "dataset_labels": {"sintel": "Sintel", "tum": "TUM-dynamics", "scannet": "ScanNet"},
        "metrics": (("ate", "ATE ↓", "min"), ("rpe_trans", "RPEtrans ↓", "min"), ("rpe_rot", "RPErot ↓", "min")),
    },
    "mv_recon": {
        "title": "三维重建评测（3D Reconstruction）",
        "datasets": ("7scenes", "NRGBD"),
        "dataset_labels": {"7scenes": "7-Scenes", "NRGBD": "NRGBD"},
        "metrics": (
            ("acc", "Acc mean ↓", "min"),
            ("acc_med", "Acc med. ↓", "min"),
            ("comp", "Comp mean ↓", "min"),
            ("comp_med", "Comp med. ↓", "min"),
            ("nc", "NC mean ↑", "max"),
            ("nc_med", "NC med. ↑", "max"),
        ),
    },
}

MODEL_LABELS = {
    "official": "官方 STream3R",
    "trained": TRAINED_MODEL_CODE,
}
MODE_LABELS = {"window": "Window=5", "full": "Full（全量）"}


def table_columns(task):
    spec = TASK_SPECS[task]
    return [
        (dataset, metric, direction, f"{spec['dataset_labels'][dataset]} {label}")
        for dataset in spec["datasets"]
        for metric, label, direction in spec["metrics"]
    ]


def value_at(report, mode, task, dataset, metric):
    values = report["results"][mode][task].get(dataset)
    return values.get(metric) if isinstance(values, dict) else None


def format_metric(value, metric):
    if not isinstance(value, (int, float)):
        return "—"
    if metric == "δ < 1.25":
        return f"{value * 100:.2f}"
    if metric in {"ate", "rpe_trans", "rpe_rot"}:
        return f"{value:.5f}"
    if metric in {"acc", "acc_med", "comp", "comp_med", "nc", "nc_med"}:
        return f"{value:.3f}"
    return f"{value:.4f}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def task_table_for_model(report, task):
    columns = table_columns(task)
    headers = ["模式"] + [column[3] for column in columns]
    rows = []
    for mode in MODES:
        row = [MODE_LABELS[mode]]
        row.extend(
            format_metric(value_at(report, mode, task, dataset, metric), metric)
            for dataset, metric, _direction, _label in columns
        )
        rows.append(row)
    return markdown_table(headers, rows)


def comparison_task_table(reports, task):
    columns = table_columns(task)
    headers = ["模型", "模式"] + [column[3] for column in columns]
    rows = []
    for mode in MODES:
        for model in ("official", "trained"):
            row = [MODEL_LABELS[model], MODE_LABELS[mode]]
            for dataset, metric, direction, _label in columns:
                value = value_at(reports[model], mode, task, dataset, metric)
                rendered = format_metric(value, metric)
                peer = value_at(
                    reports["trained" if model == "official" else "official"],
                    mode,
                    task,
                    dataset,
                    metric,
                )
                if isinstance(value, (int, float)) and isinstance(peer, (int, float)):
                    best = value <= peer if direction == "min" else value >= peer
                    if best:
                        rendered = f"**{rendered}**"
                row.append(rendered)
            rows.append(row)
    return markdown_table(headers, rows)


def task_win_counts(reports, task):
    wins = {"official": 0, "trained": 0, "tie": 0, "compared": 0}
    for mode in MODES:
        for dataset, metric, direction, _label in table_columns(task):
            a = value_at(reports["official"], mode, task, dataset, metric)
            b = value_at(reports["trained"], mode, task, dataset, metric)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            wins["compared"] += 1
            if a == b:
                wins["tie"] += 1
            elif (a < b and direction == "min") or (a > b and direction == "max"):
                wins["official"] += 1
            else:
                wins["trained"] += 1
    return wins


def context_win_counts(report):
    wins = {"full": 0, "window": 0, "tie": 0, "compared": 0}
    for task in ("video_depth", "relpose", "mv_recon"):
        for dataset, metric, direction, _label in table_columns(task):
            window = value_at(report, "window", task, dataset, metric)
            full = value_at(report, "full", task, dataset, metric)
            if not isinstance(window, (int, float)) or not isinstance(full, (int, float)):
                continue
            wins["compared"] += 1
            if window == full:
                wins["tie"] += 1
            elif (full < window and direction == "min") or (full > window and direction == "max"):
                wins["full"] += 1
            else:
                wins["window"] += 1
    return wins


def model_markdown(report):
    failed = [row for row in report["status"] if row["status"] != "OK"]
    lines = [
        f"# {MODEL_LABELS[report['model']]} 评测报告",
        "",
        "表格结构参照 STream3R 原论文：按任务组织数据集，使用 ↓/↑ 标明指标方向，`δ<1.25` 以百分数表示。",
        "",
        f"共执行 {len(report['status'])} 个评测步骤，失败 {len(failed)} 个。`Window=5` 使用五帧滑动窗口，`Full` 使用全部可用帧。",
    ]
    if report["model"] == "trained":
        lines += [
            "",
            f"> **模型代号：** `{TRAINED_MODEL_CODE}`；**模型注释：** {TRAINED_MODEL_ANNOTATION}。本地有限算力训练；训练运行 `stream3r_19`；源检查点 `best-train-009.ckpt`；epoch 9；global step 54,370；归档日期 2026-08-22；评测日期 2026-08-21 至 2026-08-22。",
        ]
    else:
        lines += ["", "> **模型注释：** STream3R 官方发布权重；评测日期 2026-08-21 至 2026-08-22。"]
    for number, task in enumerate(TASK_SPECS, start=1):
        lines += ["", f"## Table {number}. {TASK_SPECS[task]['title']}", ""]
        lines += task_table_for_model(report, task)

    context = context_win_counts(report)
    lines += [
        "",
        "## 整体总结",
        "",
        "- 单帧深度不依赖序列上下文，因此 `Window=5` 与 `Full` 两行按设计完全相同。",
        f"- 在视频深度、相机位姿和三维重建中，`Full` 在 {context['compared']} 个可比单元中有 {context['full']} 个更优，`Window=5` 有 {context['window']} 个更优。",
        "- `Full / Video Depth / Bonn` 没有汇总指标：全部 110 帧序列均超过 32 GB GPU 显存限制。报告保留为缺失值，没有用部分预测推算结果。",
    ]
    if report["model"] == "official":
        lines += [
            "- 官方权重在 Sintel/Bonn/KITTI 单帧深度、TUM/Sintel 位姿以及 NRGBD 重建上表现更稳定。",
            "- 相比五帧窗口，全量上下文明显改善相机位姿，尤其是 Sintel 和 ScanNet，但显存代价更高。",
        ]
    else:
        lines += [
            "- 训练模型改善了 NYU-v2 单帧深度，并在 ScanNet 位姿和 7-Scenes 重建上更有竞争力。",
            "- 其最大上下文收益出现在 NRGBD 重建；窗口模式的 NRGBD 结果明显较弱，表明它对上下文长度较敏感。",
        ]
    if failed:
        lines += ["", "## 失败步骤", ""]
        lines += [f"- `{row['step']}` (exit {row['exit_code']})" for row in failed]
    return "\n".join(lines) + "\n"


def flattened(report):
    return {
        (mode, task, dataset, metric): value
        for mode, task, dataset, metric, value in metric_rows(report)
        if isinstance(value, (int, float))
    }


def comparison_markdown(official, trained):
    reports = {"official": official, "trained": trained}
    lines = [
        "# STream3R 官方模型与训练模型评测总览",
        "",
        "本报告参照 STream3R 原论文的紧凑表格结构，只列出本次实际复现的结果，不把论文中的其他基线数值混入本地测量。",
        "",
        f"> **训练模型代号：** `{TRAINED_MODEL_CODE}`；**注释：** {TRAINED_MODEL_ANNOTATION}。本地有限算力训练；运行 `stream3r_19`；检查点 `best-train-009.ckpt`（epoch 9，global step 54,370）；归档日期 2026-08-22；评测日期 2026-08-21 至 2026-08-22。",
        "",
        "`Window=5` 与 `Full` 分开报告。↓ 表示越低越好，↑ 表示越高越好；粗体表示同一推理模式下两个模型中的更优值。`δ<1.25` 使用百分数。",
    ]
    for number, task in enumerate(TASK_SPECS, start=1):
        lines += ["", f"## Table {number}. {TASK_SPECS[task]['title']}", ""]
        lines += comparison_task_table(reports, task)

    counts = {task: task_win_counts(reports, task) for task in TASK_SPECS}
    official_total = sum(item["official"] for item in counts.values())
    trained_total = sum(item["trained"] for item in counts.values())
    compared_total = sum(item["compared"] for item in counts.values())
    lines += [
        "",
        "## 整体分析",
        "",
        f"四张表共有 {compared_total} 个可直接比较的指标单元：官方模型领先 {official_total} 项，训练模型领先 {trained_total} 项。该计数只用于描述覆盖面；不同指标和数据集量纲不同，不应再简单平均成一个总分。",
        "",
        f"- **单帧深度：** 官方模型 {counts['monodepth']['official']} 项领先，训练模型 {counts['monodepth']['trained']} 项领先。训练模型在 NYU-v2 上略有改善，但在 Sintel、Bonn 和 KITTI 上退化。窗口与全量数值相同是正常现象，因为每张图像独立评测。",
        f"- **视频深度：** 官方模型 {counts['video_depth']['official']} 项领先，训练模型 {counts['video_depth']['trained']} 项领先。全量模式下训练模型在 Sintel 上明显更好，官方模型在 KITTI 上仍占优；窗口模式的 Bonn 和 KITTI 也更偏向官方模型。两模型的全量 Bonn 均因 GPU OOM 缺失。",
        f"- **相机位姿：** 官方模型 {counts['relpose']['official']} 项领先，训练模型 {counts['relpose']['trained']} 项领先。官方模型主导 TUM-dynamics 和 Sintel；训练模型改善 ScanNet 的 ATE/平移误差，并在 `Window=5` 下赢得 ScanNet 三项指标，显示出更强的室内场景适应性。",
        f"- **三维重建：** 官方模型 {counts['mv_recon']['official']} 项领先，训练模型 {counts['mv_recon']['trained']} 项领先。训练模型在 7-Scenes 上稳定更强；在 NRGBD 上，窗口模式明显较弱，但全量上下文恢复显著并获得最佳 Acc，官方模型则保留更好的 Comp 和 NC。",
        "- **上下文长度权衡：** 全量上下文通常有利于两个模型，位姿和重建任务最明显。`Window=5` 是显存恒定、实际更易部署的方案；`Full` 在多个长程任务上精度更高，但长序列可能超过 32 GB 显存。",
        "",
        "### 结论",
        "",
        "训练模型并不能全面替代官方模型。它在 NYU-v2、ScanNet、7-Scenes 以及全量 Sintel 视频深度上取得收益，但在多个深度数据集和动态位姿基准上的跨域稳健性下降。实际部署时，如果目标场景接近其提升明显的室内数据分布，可以优先验证训练模型；面向未知或更广泛场景时，官方模型仍是更稳妥的通用基线。",
        "",
        "表格格式参考：[STream3R 原论文（Tables 1–4）](https://arxiv.org/abs/2508.10893)。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root
    reports = {}
    model_dirs = {"official": "official", "trained": TRAINED_MODEL_CODE}
    for name, directory in model_dirs.items():
        model_dir = root / directory
        report = collect_model(model_dir, name)
        reports[name] = report
        (model_dir / f"{name}_model_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        (model_dir / f"{name}_model_report.md").write_text(model_markdown(report))
    (root / "comparison_report.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n"
    )
    (root / "comparison_report.md").write_text(
        comparison_markdown(reports["official"], reports["trained"])
    )


if __name__ == "__main__":
    main()
