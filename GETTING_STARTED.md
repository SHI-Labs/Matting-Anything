# Getting Started with MAM
This doc provides the instructions to reproduce the MAM.

## Data Preparation

- During training, we used foregrounds from AIM, Distinctions-646, AM2K, Human-2K, and RefMatte to ensure a diverse range of instance classes. We used COCO and BG20K to provide a mix of both real-world and synthetic backgrounds.

- During evaluation, we tested MAM on a variety of image matting benchmarks including the semantic image matting benchmarks
PPM-100, AM2K, PM-10K, the instance image matting benchmark RWP636, HIM2K, and the referring image matting benchmark RefMatte-RW100.

## Training MAM
- Please prepare all these datasets and specify the paths of these datasets in the config file.

- Setup the environment and install MAM following the instructions in the [INSTALL.md](INSTALL.md).

- Train MAM with SAM ViT-B checkpoint and 8 GPUs
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py --config config/MAM-ViTB-8gpu.toml
```

- Train MAM with SAM ViT-L checkpoint and 8 GPUs
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py --config config/MAM-ViTL-8gpu.toml
```

- Train MAM with SAM ViT-H checkpoint and 8 GPUs
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py --config config/MAM-ViTH-8gpu.toml
```

## Evaluation

- Please prepare all these test sets of benchmarks and specify the paths of them in the config file.

- Setup the environment and install MAM following the instructions in the [INSTALL.md](INSTALL.md).

- Evaluate MAM based on SAM ViT-B checkpoint on each benchmark
    - PPM-100
    ```bash
    python inference_mam.py --config config/MAM-ViTB-8gpu.toml --checkpoint checkpoints/mam_vitb.pth --benchmark ppm100 --output outputs/ppm100 --alphaguide

    python evaluation/evaluation_ppm100.py --pred-dir outputs/ppm100
    ```
    - AM2K
    ```bash
    python inference_mam.py --config config/MAM-ViTB-8gpu.toml --checkpoint checkpoints/mam_vitb.pth --benchmark am2k --output outputs/am2k --alphaguide

    python evaluation/evaluation_am2k.py --pred-dir outputs/am2k
    ```
    - PM-10K
    ```bash
    python inference_mam.py --config config/MAM-ViTB-8gpu.toml --checkpoint checkpoints/mam_vitb.pth --benchmark pm10k --output outputs/pm10k --alphaguide

    python evaluation/evaluation_pm10k.py --pred-dir outputs/pm10k
    ```
    - RWP636
    ```bash
    python inference_mam.py --config config/MAM-ViTB-8gpu.toml --checkpoint checkpoints/mam_vitb.pth --benchmark rwp636 --output outputs/rwp636 --alphaguide

    python evaluation/IMQ_quick_rwp.py path/to/outputs/rwp636 path/to/RealWorldPortrait-636/alpha
    ```
    - HIM2K
    ```bash
    python inference_mam.py --config config/MAM-ViTB-8gpu.toml --checkpoint checkpoints/mam_vitb.pth --benchmark him2k --output outputs/him2k/ --maskguide

    python evaluation/IMQ.py path/to/outputs/him2k path/to/HIM2K/alphas/natural/

    python inference_mam.py --config config/MAM-ViTB-8gpu.toml --checkpoint checkpoints/mam_vitb.pth --benchmark him2k_comp --output outputs/him2k_comp --maskguide

    python evaluation/IMQ.py path/to/outputs/him2k_comp path/to/HIM2K/alphas/comp/
    ```
    - RefMatte-RW100
    ```bash
    python inference_mam.py --config config/MAM-ViTB-8gpu.toml --checkpoint checkpoints/mam_vitb.pth --benchmark rw100 --output outputs/rw100 --maskguide --prompt text/box/point

    python evaluation/evaluation_refmatte.py --pred-dir outputs/rw100
    ```


