#!/usr/bin/env python3
"""Full-volume OASIS/Learn2Reg evaluation for SpikeReg checkpoints."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from examples.oasis_dataset import L2RPKLDataset, OASISL2RDataset
from SpikeReg.models import PretrainedUNet, SpikeRegUNet
from SpikeReg.utils.metrics import (
    jacobian_determinant_stats,
    labelwise_dice_score,
    labelwise_hd95,
    normalized_cross_correlation,
)
from SpikeReg.utils.warping import SpatialTransformer


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def _load_config(args, checkpoint: Dict) -> Dict:
    if args.config:
        with open(args.config, "r") as f:
            return yaml.safe_load(f)
    return checkpoint.get("config", {})


def _build_model(config: Dict, checkpoint: Dict, model_type: str, device: torch.device):
    phase = checkpoint.get("training_phase", "")
    if model_type == "auto":
        model_type = "ann" if phase == "pretrain" else "snn"

    model_config = config.get("model", config)
    if model_type == "ann":
        model = PretrainedUNet(model_config)
    elif model_type == "snn":
        model = SpikeRegUNet(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = _strip_module_prefix(checkpoint.get("model_state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[evaluate_oasis] missing keys: {missing}")
        print(f"[evaluate_oasis] unexpected keys: {unexpected}")
    return model.to(device).eval(), model_type


def _resolve_data_root(args, config: Dict) -> str:
    if args.data_root:
        return args.data_root
    data_cfg = config.get("data", {})
    train_dir = data_cfg.get("train_dir")
    if not train_dir:
        raise ValueError("Pass --data-root or set data.train_dir in the config/checkpoint.")
    return train_dir


def _safe_mean(values: List[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_std(values: List[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.std(finite)) if finite else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpikeReg on full-volume OASIS/Learn2Reg val pairs.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--config", help="Optional YAML config. Defaults to checkpoint config.")
    parser.add_argument("--data-root", help="L2R OASIS root containing imagesTr/, labelsTr/, and OASIS_dataset.json")
    parser.add_argument("--dataset-format", choices=["auto", "nii", "pkl"], default="auto")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--model-type", choices=["auto", "ann", "snn"], default="auto")
    parser.add_argument("--target-shape", nargs=3, type=int, help="Override evaluation shape, e.g. 128 128 128")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit number of validation pairs for smoke tests")
    parser.add_argument("--skip-hd95", action="store_true", help="Skip HD95 computation")
    parser.add_argument("--output-json", help="Write aggregate and per-pair metrics to JSON")
    parser.add_argument("--output-csv", help="Write per-pair metrics to CSV")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = _load_config(args, checkpoint)
    model, model_type = _build_model(config, checkpoint, args.model_type, device)

    data_root = _resolve_data_root(args, config)
    if args.target_shape:
        target_shape = tuple(args.target_shape)
    else:
        target_shape = tuple(config.get("data", {}).get("resample_size", [128, 128, 128]))

    dataset_format = args.dataset_format
    if dataset_format == "auto":
        dataset_format = str(config.get("data", {}).get("dataset_format", "nii")).lower()
        dataset_format = "pkl" if dataset_format in {"pkl", "l2r_pkl"} else "nii"

    if dataset_format == "pkl":
        val_root = args.data_root or config.get("data", {}).get("val_dir") or data_root
        dataset = L2RPKLDataset(
            data_path=val_root,
            split="val",
            input_dim=target_shape,
            max_pairs=args.max_pairs,
            use_labels=True,
        )
        data_root = val_root
    else:
        dataset = OASISL2RDataset(
            data_path=data_root,
            split="val",
            input_dim=target_shape,
            is_pair=True,
            max_pairs=args.max_pairs,
            use_labels=True,
        )
    transformer = SpatialTransformer().to(device)
    seg_transformer = SpatialTransformer(mode="nearest", padding_mode="border").to(device)

    pair_rows = []
    dice_values = []
    hd95_values = []
    ncc_values = []
    jac_neg_values = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating OASIS pairs"):
            batch = dataset[idx]
            fixed = batch["fixed"].unsqueeze(0).to(device)
            moving = batch["moving"].unsqueeze(0).to(device)
            fixed_seg = batch["segmentation_fixed"].unsqueeze(0).to(device)
            moving_seg = batch["segmentation_moving"].unsqueeze(0).to(device)

            if model_type == "ann":
                displacement = model(fixed, moving)
            else:
                output = model(fixed, moving)
                displacement = output["displacement"]

            warped = transformer(moving, displacement)
            warped_seg = seg_transformer(moving_seg.float(), displacement).long()

            dice = labelwise_dice_score(
                warped_seg,
                fixed_seg,
                label_ids=dataset.eval_label_ids,
            )
            mean_dice = float(torch.nanmean(dice).item()) if dice.numel() else float("nan")
            jac_stats = jacobian_determinant_stats(displacement)
            ncc = float(normalized_cross_correlation(fixed, warped).mean().item())

            row = {
                "pair_index": idx,
                "dice": mean_dice,
                "ncc": ncc,
                "jacobian_negative_fraction": float(jac_stats["negative_fraction"]),
                "jacobian_folding_fraction": float(jac_stats["folding_fraction"]),
            }

            if not args.skip_hd95:
                hd95_by_label = labelwise_hd95(
                    warped_seg.squeeze(0),
                    fixed_seg.squeeze(0),
                    label_ids=dataset.eval_label_ids,
                )
                row["hd95"] = _safe_mean(list(hd95_by_label.values()))
                row["hd95_num_labels"] = len(hd95_by_label)
                hd95_values.append(row["hd95"])

            pair_rows.append(row)
            dice_values.append(mean_dice)
            ncc_values.append(ncc)
            jac_neg_values.append(row["jacobian_negative_fraction"])

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_type": model_type,
        "data_root": str(Path(data_root).resolve()),
        "dataset_format": dataset_format,
        "target_shape": list(target_shape),
        "num_pairs": len(pair_rows),
        "dice_mean": _safe_mean(dice_values),
        "dice_std": _safe_std(dice_values),
        "ncc_mean": _safe_mean(ncc_values),
        "ncc_std": _safe_std(ncc_values),
        "jacobian_negative_fraction_mean": _safe_mean(jac_neg_values),
        "jacobian_negative_fraction_std": _safe_std(jac_neg_values),
    }
    if hd95_values:
        summary["hd95_mean"] = _safe_mean(hd95_values)
        summary["hd95_std"] = _safe_std(hd95_values)

    print(json.dumps(summary, indent=2))

    if args.output_json:
        payload = {"summary": summary, "pairs": pair_rows}
        Path(args.output_json).write_text(json.dumps(payload, indent=2))

    if args.output_csv and pair_rows:
        keys = sorted({k for row in pair_rows for k in row.keys()})
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(pair_rows)


if __name__ == "__main__":
    main()
