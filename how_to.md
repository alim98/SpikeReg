# SpikeReg on Viper-GPU

## Launch training

```bash
cd /viper/u2/almik/SpikeReg
bash launch_spikereg_training.sh
```

Outputs → `/nexus/posix0/MBR-neuralsystems/alim/experiments3/SR/runs/<job_id>/`

## TensorBoard

Run on a login node (inside tmux):

```bash
/ptmp/almik/cephclr_venv_rocm63/bin/tensorboard \
  --logdir /nexus/posix0/MBR-neuralsystems/alim/experiments3/SR/runs \
  --port 6006 --bind_all
```

Then forward from your laptop:

```bash
ssh -L 6006:viper12i.mpcdf.mpg.de:6006 gate.mpcdf.mpg.de
```

Open → http://localhost:6006

## Resume from checkpoint

```bash
bash launch_spikereg_training.sh --start_from_checkpoint /path/to/checkpoint.pth
```

## Override nodes / QOS

```bash
# e.g. drop back to 2 nodes for a quick test
#SBATCH --nodes=2 is in train_spikereg.sh
```


abblations:
scripts/run_spikereg_paper_suite.sh --mode evaluate-submit
scripts/run_spikereg_paper_suite.sh --mode ablate
scripts/run_spikereg_paper_suite.sh --mode ablate --submit

