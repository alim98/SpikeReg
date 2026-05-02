#!/usr/bin/env python3
"""Full-volume OASIS/Learn2Reg evaluation for SpikeReg checkpoints."""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from examples.oasis_dataset import L2RPKLDataset, OASISL2RDataset
from SpikeReg.models import PretrainedUNet, SpikeRegUNet, convert_pretrained_to_spiking
from SpikeReg.utils.metrics import (
    compute_energy_estimate,
    jacobian_determinant,
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


def _load_pretrained_ann(config: Dict, checkpoint: Dict, device: torch.device) -> PretrainedUNet:
    model = PretrainedUNet(config.get("model", config))
    state = _strip_module_prefix(checkpoint.get("model_state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[evaluate_oasis] ANN missing keys before conversion: {missing}")
        print(f"[evaluate_oasis] ANN unexpected keys before conversion: {unexpected}")
    return model.to(device).eval()


def _build_converted_snn(
    config: Dict,
    checkpoint: Dict,
    dataset,
    device: torch.device,
    calibration_pairs: int,
):
    ann = _load_pretrained_ann(config, checkpoint, device)
    conversion_cfg = config.get("conversion", {})
    threshold_percentile = float(conversion_cfg.get("threshold_percentile", 99.0))
    skip_calibration = bool(conversion_cfg.get("skip_calibration", False)) or threshold_percentile <= 0
    calibration_loader = None

    if not skip_calibration:
        requested_pairs = int(calibration_pairs or conversion_cfg.get("calibration_samples", 3))
        n_cal = max(1, min(requested_pairs, len(dataset)))
        calibration_loader = DataLoader(
            Subset(dataset, list(range(n_cal))),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        print(
            f"[evaluate_oasis] converting ANN to SNN with threshold_percentile="
            f"{threshold_percentile}, calibration_pairs={n_cal}"
        )
    else:
        print("[evaluate_oasis] converting ANN to SNN without threshold calibration")

    model = convert_pretrained_to_spiking(
        ann,
        config.get("model", config),
        threshold_percentile=threshold_percentile,
        calibration_loader=calibration_loader,
    )
    return model.to(device).eval(), "snn"


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


def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value, dtype=np.float64).mean())
    return float(value)


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _time_steps_from_config(config: Dict) -> int:
    model_cfg = config.get("model", config)
    candidates = [model_cfg.get("input_time_window", 4)]
    candidates.extend(model_cfg.get("encoder_time_windows", []) or [])
    candidates.extend(model_cfg.get("decoder_time_windows", []) or [])
    return int(max(int(v) for v in candidates))


def _sdlogj(displacement: torch.Tensor) -> float:
    det = jacobian_determinant(displacement)
    positive = det[det > 1e-6]
    if positive.numel() < 2:
        return float("nan")
    return float(torch.log(positive).std().item())


def _mean_dict(values: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: _safe_mean(v) for k, v in sorted(values.items())}


def _sum_dict(values: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.nansum(v)) for k, v in sorted(values.items())}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpikeReg on full-volume OASIS/Learn2Reg val pairs.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--config", help="Optional YAML config. Defaults to checkpoint config.")
    parser.add_argument("--data-root", help="L2R OASIS root containing imagesTr/, labelsTr/, and OASIS_dataset.json")
    parser.add_argument("--dataset-format", choices=["auto", "nii", "pkl"], default="auto")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--model-type", choices=["auto", "ann", "snn"], default="auto")
    parser.add_argument("--convert-ann-to-snn", action="store_true", help="Treat --checkpoint as an ANN checkpoint, convert with --config, then evaluate the converted SNN without training")
    parser.add_argument("--calibration-pairs", type=int, default=3, help="Number of eval pairs to use for conversion calibration in --convert-ann-to-snn mode")
    parser.add_argument("--target-shape", nargs=3, type=int, help="Override evaluation shape, e.g. 128 128 128")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit number of validation pairs for smoke tests")
    parser.add_argument("--skip-hd95", action="store_true", help="Skip HD95 computation")
    parser.add_argument("--output-json", help="Write aggregate and per-pair metrics to JSON")
    parser.add_argument("--output-csv", help="Write per-pair metrics to CSV")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = _load_config(args, checkpoint)
    model_config = config.get("model", config)
    time_steps = _time_steps_from_config(config)

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

    if args.convert_ann_to_snn:
        model, model_type = _build_converted_snn(
            config,
            checkpoint,
            dataset,
            device,
            args.calibration_pairs,
        )
    else:
        model, model_type = _build_model(config, checkpoint, args.model_type, device)

    transformer = SpatialTransformer().to(device)
    seg_transformer = SpatialTransformer(mode="nearest", padding_mode="border").to(device)

    pair_rows = []
    dice_values = []
    hd95_values = []
    ncc_values = []
    jac_neg_values = []
    jac_fold_values = []
    sdlogj_values = []
    inference_time_values = []
    displacement_mean_values = []
    displacement_max_values = []
    spike_rate_values = defaultdict(list)
    spike_count_values = defaultdict(list)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating OASIS pairs"):
            batch = dataset[idx]
            fixed = batch["fixed"].unsqueeze(0).to(device)
            moving = batch["moving"].unsqueeze(0).to(device)
            fixed_seg = batch["segmentation_fixed"].unsqueeze(0).to(device)
            moving_seg = batch["segmentation_moving"].unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            spike_rates = {}
            spike_counts_number = {}
            if model_type == "ann":
                displacement = model(fixed, moving)
            else:
                output = model(fixed, moving)
                displacement = output["displacement"]
                spike_rates = {k: _to_float(v) for k, v in output.get("spike_counts", {}).items()}
                spike_counts_number = {
                    k: _to_float(v) for k, v in output.get("spike_counts_number", {}).items()
                }

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_time = time.perf_counter() - start_time

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
            disp_norm = torch.linalg.norm(displacement.detach(), dim=1)
            disp_mean = float(disp_norm.mean().item())
            disp_max = float(disp_norm.max().item())
            sdlogj = _sdlogj(displacement)

            row = {
                "pair_index": idx,
                "dice": mean_dice,
                "ncc": ncc,
                "jacobian_negative_fraction": float(jac_stats["negative_fraction"]),
                "jacobian_folding_fraction": float(jac_stats["folding_fraction"]),
                "jacobian_mean": float(jac_stats["mean"]),
                "jacobian_std": float(jac_stats["std"]),
                "sdlogj": sdlogj,
                "inference_time_sec": float(inference_time),
                "displacement_mean_norm": disp_mean,
                "displacement_max_norm": disp_max,
            }

            if spike_rates:
                row["spike_rates"] = spike_rates
                row["spike_counts_number"] = spike_counts_number
                row["mean_spike_rate"] = _safe_mean(list(spike_rates.values()))
                row["total_spike_count"] = float(np.nansum(list(spike_counts_number.values())))
                energy = compute_energy_estimate(
                    spike_rates,
                    volume_shape=target_shape,
                    encoder_channels=model_config.get("encoder_channels"),
                    decoder_channels=model_config.get("decoder_channels"),
                    time_steps=time_steps,
                )
                row.update({f"energy_{k}": float(v) for k, v in energy.items()})
                for layer, value in spike_rates.items():
                    spike_rate_values[layer].append(value)
                for layer, value in spike_counts_number.items():
                    spike_count_values[layer].append(value)

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
            jac_fold_values.append(row["jacobian_folding_fraction"])
            sdlogj_values.append(sdlogj)
            inference_time_values.append(float(inference_time))
            displacement_mean_values.append(disp_mean)
            displacement_max_values.append(disp_max)

    reference_energy = compute_energy_estimate(
        {"reference": 0.1},
        volume_shape=target_shape,
        encoder_channels=model_config.get("encoder_channels"),
        decoder_channels=model_config.get("decoder_channels"),
        time_steps=time_steps,
    )
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
        "jacobian_folding_fraction_mean": _safe_mean(jac_fold_values),
        "jacobian_folding_fraction_std": _safe_std(jac_fold_values),
        "sdlogj_mean": _safe_mean(sdlogj_values),
        "sdlogj_std": _safe_std(sdlogj_values),
        "inference_time_sec_mean": _safe_mean(inference_time_values),
        "inference_time_sec_std": _safe_std(inference_time_values),
        "parameter_count": _parameter_count(model),
        "displacement_mean_norm_mean": _safe_mean(displacement_mean_values),
        "displacement_max_norm_max": float(np.nanmax(displacement_max_values)) if displacement_max_values else float("nan"),
        "time_steps": time_steps,
        "ann_macs_G": float(reference_energy["ann_macs_G"]),
        "E_ANN_mJ": float(reference_energy["E_ANN_mJ"]),
    }

    if spike_rate_values:
        mean_spike_rates = _mean_dict(spike_rate_values)
        total_spike_counts = _sum_dict(spike_count_values)
        energy = compute_energy_estimate(
            mean_spike_rates,
            volume_shape=target_shape,
            encoder_channels=model_config.get("encoder_channels"),
            decoder_channels=model_config.get("decoder_channels"),
            time_steps=time_steps,
        )
        summary["spike_rates_mean_by_layer"] = mean_spike_rates
        summary["spike_counts_total_by_layer"] = total_spike_counts
        summary["mean_spike_rate"] = float(energy["mean_spike_rate"])
        summary["snn_acs_G"] = float(energy["snn_acs_G"])
        summary["E_SNN_mJ"] = float(energy["E_SNN_mJ"])
        summary["energy_ratio_snn_over_ann"] = float(energy["energy_ratio_snn_over_ann"])
        summary["energy_reduction_factor"] = float(energy["energy_reduction_factor"])
    else:
        summary["energy_ratio_snn_over_ann"] = 1.0
        summary["energy_reduction_factor"] = 1.0
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
