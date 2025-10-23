from SpikeReg.inference import *

def main():
    get_inference()



if __name__ == '__main__':
    main() 

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from SpikeReg.models import SpikeRegUNet
from SpikeReg.registration import IterativeRegistration
from SpikeReg.losses import SpikeRegLoss
from SpikeReg.utils.metrics import compute_registration_metrics

class DummyVolDataset(Dataset):
    def __init__(self, n=6, shape=(32,32,32), seed=0):
        rng = np.random.RandomState(seed)
        fixed = rng.rand(n, *shape).astype(np.float32)
        disp = rng.normal(0.0, 0.03, size=(n, 3, *shape)).astype(np.float32)
        moving = np.clip(fixed + disp.sum(1), 0.0, 1.0).astype(np.float32)
        self.fixed = fixed
        self.moving = moving
    def __len__(self):
        return len(self.fixed)
    def __getitem__(self, idx):
        f = torch.from_numpy(self.fixed[idx]).unsqueeze(0)
        m = torch.from_numpy(self.moving[idx]).unsqueeze(0)
        return {"fixed": f, "moving": m}

def cfg():
    return {
        "model": {
            "patch_size": 32,
            "in_channels": 2,
            "base_channels": 8,
            "encoder_channels": [8,16,32],
            "decoder_channels": [16,8,8],
            "encoder_time_windows": [2,2,2],
            "decoder_time_windows": [2,2,2],
            "encoder_tau_u": [0.9,0.85,0.8],
            "decoder_tau_u": [0.8,0.85,0.9],
            "skip_merge": ["concatenate","average","concatenate"],
            "displacement_scale": 1.0,
            "input_time_window": 3
        },
        "train": {
            "epochs": 3,
            "lr": 1e-3,
            "batch_size": 2,
            "num_iterations": 2
        },
        "loss": {
            "similarity_type": "ncc",
            "similarity_weight": 1.0,
            "regularization_type": "bending",
            "regularization_weight": 1e-3,
            "spike_weight": 1e-2,
            "spike_balance_weight": 1e-2,
            "target_spike_rate": 0.1
        }
    }

def main2():
    c = cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DummyVolDataset(n=10, seed=1)
    val_ds = DummyVolDataset(n=4, seed=2)
    train_loader = DataLoader(train_ds, batch_size=c["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=c["train"]["batch_size"], shuffle=False)
    model = SpikeRegUNet(c["model"]).to(device)
    reg = IterativeRegistration(model, num_iterations=c["train"]["num_iterations"], early_stop_threshold=1e-4)
    optim = torch.optim.AdamW(model.parameters(), lr=c["train"]["lr"])
    criterion = SpikeRegLoss(
        similarity_type=c["loss"]["similarity_type"],
        similarity_weight=c["loss"]["similarity_weight"],
        regularization_type=c["loss"]["regularization_type"],
        regularization_weight=c["loss"]["regularization_weight"],
        spike_weight=c["loss"]["spike_weight"],
        spike_balance_weight=c["loss"]["spike_balance_weight"],
        target_spike_rate=c["loss"]["target_spike_rate"]
    )
    for epoch in range(1, c["train"]["epochs"]+1):
        model.train()
        totals = {"total":0.0,"similarity":0.0,"regularization":0.0,"spike":0.0}
        steps = 0
        for batch in train_loader:
            fixed = batch["fixed"].to(device)
            moving = batch["moving"].to(device)
            optim.zero_grad()
            out = reg(fixed, moving, return_all_iterations=True)
            disp = out["displacement"]
            warped = out["warped"]
            spike_counts = out["spike_count_history"][-1] if len(out.get("spike_count_history",[]))>0 else {}
            loss, parts = criterion(fixed, moving, disp, warped, spike_counts)
            loss.backward()
            optim.step()
            totals["total"] += float(parts["total"])
            totals["similarity"] += float(parts["similarity"])
            totals["regularization"] += float(parts["regularization"])
            totals["spike"] += float(parts["spike"])
            steps += 1
        avg = {k: v/max(1,steps) for k,v in totals.items()}
        print(f"epoch={epoch} train_total={avg['total']:.4f} sim={avg['similarity']:.4f} reg={avg['regularization']:.6f} spike={avg['spike']:.6f}")
        model.eval()
        with torch.no_grad():
            v_tot = {"ncc":0.0,"mse":0.0,"ssim":0.0,"jac_neg":0.0}
            v_steps = 0
            for batch in val_loader:
                fixed = batch["fixed"].to(device)
                moving = batch["moving"].to(device)
                out = reg(fixed, moving, return_all_iterations=False)
                disp = out["displacement"]
                warped = out["warped"]
                m = compute_registration_metrics(fixed, moving, warped, disp)
                v_tot["ncc"] += m["ncc"]; v_tot["mse"] += m["mse"]; v_tot["ssim"] += m["ssim"]; v_tot["jac_neg"] += m["jacobian_negative_fraction"]
                v_steps += 1
            v_avg = {k: v/max(1,v_steps) for k,v in v_tot.items()}
            print(f"epoch={epoch} val_ncc={v_avg['ncc']:.4f} val_mse={v_avg['mse']:.4f} val_ssim={v_avg['ssim']:.4f} val_jac_neg={v_avg['jac_neg']:.4f}")

# if __name__ == "__main__":
#     main()