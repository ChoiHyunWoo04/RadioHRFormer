# RadioHRFormer

Official PyTorch implementation of **RadioHRFormer**, a high-resolution Transformer framework for sampling-free radio-map prediction with propagation-aware pretraining.

RadioHRFormer is built on HRFormer and is adapted for dense radio-map regression on **RadioMapSeer**. The model preserves a high-resolution representation throughout the backbone, repeatedly exchanges information across multiple resolutions, and uses stem skip connections in the decoder to recover full-resolution radio maps.

The repository supports:

- DPM radio-map prediction
- carsDPM radio-map prediction
- propagation-aware pretraining with a precomputed obstacle-transmittance target
- zero-shot evaluation and fine-tuning on IRT4 / carsIRT4
- qualitative visualization of propagation-aware targets
- MAE, MSE, RMSE, NMSE, PSNR, and SSIM evaluation
- spatial Boundary / LoS / NLoS / Long-range NLoS analysis
- resolution-scalability benchmarking for runtime, memory, and FLOPs

## Repository Structure

```text
RadioHRFormer/
├── configs/
│   ├── dpm_pretrain.json
│   ├── dpm_downstream.json
│   ├── carsdpm_pretrain.json
│   ├── carsdpm_downstream.json
│   └── carsirt4_finetuning.json
├── datasets/
│   ├── physics_targets.py
│   └── rms_dataset.py
├── models/
│   ├── hrformer/
│   ├── utils/
│   ├── decoder.py
│   └── hrformer_regressor.py
├── tools/
│   ├── precompute_obstacle_targets.py
│   ├── visualize_physics_targets.py
│   ├── evaluate.py
│   ├── benchmark_runtime.py
│   └── benchmark_flops.py
├── losses.py
├── metrics.py
├── pretrain.py
├── train.py
├── utils.py
├── requirements.txt
└── LICENSE
```

## Installation

Clone the repository and create a Python environment.

```bash
git clone https://github.com/ChoiHyunWoo04/RadioHRFormer.git
cd RadioHRFormer

conda create -n radiohrformer python=3.11.15 -y
conda activate radiohrformer
```

Install PyTorch 2.4.1 with CUDA 11.8:

```bash
python -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The experiments in this repository were developed with **Python 3.11.15**, **PyTorch 2.4.1**, and **CUDA 11.8**. If a different CUDA version is used, install the corresponding PyTorch build before installing the remaining packages.

## Dataset

Download **RadioMapSeer** and place it under:

```text
./data/radiomapseer/
```

The code expects the following relevant directories:

```text
data/radiomapseer/
├── png/
│   ├── buildings_complete/
│   ├── cars/
│   └── antennas/
└── gain/
    ├── DPM/
    ├── carsDPM/
    ├── IRT4/
    └── carsIRT4/
```

The default split follows the map-level RadioMapSeer split used in this project: 550 maps for training, 50 for validation, and 100 for testing, with 80 transmitter locations per map for DPM/carsDPM. IRT4/carsIRT4 uses the two available transmitter locations per map.

## 1. Precompute Propagation-Aware Targets

RadioHRFormer pretraining uses a precomputed obstacle-transmittance map. The same precomputed maps are also used for LoS/NLoS-based spatial evaluation and target visualization.

### DPM

```bash
python tools/precompute_obstacle_targets.py \
    --config-path ./configs/dpm_pretrain.json
```

### carsDPM

```bash
python tools/precompute_obstacle_targets.py \
    --config-path ./configs/carsdpm_pretrain.json
```

By default, the generated files are stored under:

```text
./data/precomputed_obstacle/
├── building_DPM/
│   ├── train/
│   ├── val/
│   └── test/
└── cars_carsDPM/
    ├── train/
    ├── val/
    └── test/
```

Each `.pt` file stores the obstacle-transmittance target

```text
obstacle_saturating_a007 = exp(-0.07 * A(p))
```

where `A(p)` is the accumulated obstacle length along the Tx-to-pixel path.

## 2. Visualize Propagation-Aware Targets

After target precomputation, the propagation-aware maps can be visualized before training.

### DPM

```bash
python tools/visualize_physics_targets.py \
    --config-path ./configs/dpm_pretrain.json
```

### carsDPM

```bash
python tools/visualize_physics_targets.py \
    --config-path ./configs/carsdpm_pretrain.json
```

The visualizer saves the downstream radio-map label together with the Tx-centered radial prior, LoS map, and obstacle-transmittance map. The default output directory is:

```text
./save/visual_pretrain_targets/
```

Example output layout:

```text
save/visual_pretrain_targets/
└── sample_000_<map>_<tx>/
    ├── pretrain_targets_2x2.png
    ├── downstream_label.png
    ├── radial_gain.png
    ├── los.png
    └── obstacle_saturating_a007.png
```

Visualization options such as the split, number of samples, output directory, and DPI are configured in the `visualize` block of the corresponding pretraining JSON file.

## 3. Propagation-Aware Pretraining

### DPM

```bash
python pretrain.py \
    --config-path ./configs/dpm_pretrain.json \
    --save-root ./save_pretrain
```

The default checkpoint path is:

```text
./save_pretrain/dpm/weight/best.pth
```

### carsDPM

```bash
python pretrain.py \
    --config-path ./configs/carsdpm_pretrain.json \
    --save-root ./save_pretrain
```

The default checkpoint path is:

```text
./save_pretrain/carsdpm/weight/best.pth
```

The output directory also contains `last.pth`, `log.txt`, `loss.png`, and the resolved `config.json`.

## 4. Downstream Training

### DPM with propagation-aware initialization

```bash
python train.py \
    --config-path ./configs/dpm_downstream.json \
    --physics-pretrained ./save_pretrain/dpm/weight/best.pth \
    --save-root ./save
```

### carsDPM with propagation-aware initialization

```bash
python train.py \
    --config-path ./configs/carsdpm_downstream.json \
    --physics-pretrained ./save_pretrain/carsdpm/weight/best.pth \
    --save-root ./save
```

The default downstream checkpoints are stored at:

```text
./save/dpm/weight/best.pth
./save/carsdpm/weight/best.pth
```

Use `--run-name <name>` when a separate experiment directory is desired.

## 5. Evaluation

The evaluator reports:

- MAE
- MSE
- RMSE
- NMSE
- PSNR
- SSIM
- Boundary RMSE
- LoS RMSE
- NLoS RMSE
- Long-range NLoS RMSE
- parameter count and model size
- forward inference time
- CUDA memory statistics

The spatial metrics require the precomputed obstacle-transmittance maps from Step 1.

### carsDPM test evaluation

```bash
python tools/evaluate.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/carsdpm/weight/best.pth \
    --split test \
    --save-root ./save_eval \
    --run-name carsdpm_test \
    --save-pred \
    --save-gt \
    --max-save 100
```

### DPM test evaluation

```bash
python tools/evaluate.py \
    --config-path ./configs/dpm_downstream.json \
    --weight-path ./save/dpm/weight/best.pth \
    --split test \
    --save-root ./save_eval \
    --run-name dpm_test
```

Prediction PNGs use the qualitative RadioMapSeer-style rendering used in this project:

- radio field: black-to-yellow
- buildings: blue
- cars: red

Use `--no-save-pred`, `--no-save-npy`, `--no-save-error`, `--no-save-gt`, or `--no-save-gt-npy` to disable individual outputs.

## 6. IRT4 / carsIRT4 Evaluation

### carsIRT4 zero-shot evaluation

```bash
python tools/evaluate.py \
    --config-path ./configs/carsirt4_finetuning.json \
    --weight-path ./save/carsdpm/weight/best.pth \
    --split test \
    --save-root ./save_eval \
    --run-name carsirt4_zero_shot
```

### carsIRT4 fine-tuning

```bash
python train.py \
    --config-path ./configs/carsirt4_finetuning.json \
    --carsdpm-pretrained ./save/carsdpm/weight/best.pth \
    --eval-split val \
    --save-root ./save \
    --run-name carsirt4_finetuning
```

Evaluate the fine-tuned model:

```bash
python tools/evaluate.py \
    --config-path ./configs/carsirt4_finetuning.json \
    --weight-path ./save/carsirt4_finetuning/weight/best.pth \
    --split test \
    --save-root ./save_eval \
    --run-name carsirt4_finetuned
```

## 7. Resolution-Scalability Benchmark

The benchmark scripts use synthetic three-channel inputs to isolate architectural scaling from dataset-specific accuracy.

### Runtime and CUDA memory

```bash
python tools/benchmark_runtime.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/carsdpm/weight/best.pth \
    --device cuda:0 \
    --resolutions 256 512 768 1024 \
    --batch-size 1 \
    --warmup 20 \
    --repeats 100 \
    --precision fp32 \
    --cooldown-seconds 2 \
    --output-dir ./runs/radiohrformer_resolution_runtime
```

### FLOPs

```bash
python tools/benchmark_flops.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/carsdpm/weight/best.pth \
    --device cuda:0 \
    --resolutions 256 512 768 1024 \
    --batch-size 1 \
    --precision fp32 \
    --output-dir ./runs/radiohrformer_resolution_flops
```

Runtime and FLOPs are intentionally measured in separate processes so that profiler overhead does not contaminate latency measurements.

## Configuration

The main experimental settings are controlled by JSON files in `configs/`.

Common fields include:

- `runtime.gpus`: GPU indices used for training/evaluation
- `runtime.amp`: automatic mixed precision
- `data.root_dir`: RadioMapSeer root directory
- `data.target_type`: `DPM`, `carsDPM`, `IRT4`, or `carsIRT4`
- `data.batch_size`: training/validation batch size
- `model.backbone.extra`: HRFormer stage configuration
- `model.decoder`: radio-map decoder configuration
- `pretrain`: pretraining optimization settings
- `train`: downstream optimization settings
- `physics`: propagation-aware target settings
- `visualize`: target-visualization settings

## Acknowledgements

RadioHRFormer is built on the open-source **HRFormer** implementation and uses the **RadioMapSeer** dataset. We thank the authors of these projects for making their code and data publicly available.

- HRFormer: Y. Yuan *et al.*, *HRFormer: High-Resolution Transformer for Dense Prediction*, NeurIPS 2021.
- RadioMapSeer / RadioUNet: R. Levie *et al.*, *RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks*.

The HRFormer-derived source files retain their original copyright and license notices, together with notes identifying RadioHRFormer-specific modifications.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Citation

Citation information will be added after publication.
