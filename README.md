# visualization
python tools/visualize_physics_targets.py --config-path ./configs/carsdpm_pretrain.json


# precompute
python tools/precompute_obstacle_targets.py \
  --config-path ./configs/dpm_pretrain.json

python tools/precompute_obstacle_targets.py \
  --config-path ./configs/carsdpm_pretrain.json


# pretrain
python pretrain.py \
  --config-path ./configs/carsdpm_pretrain.json \
  --save-root ./save_pretrain

# train with pretrained weights
python train.py \
  --config-path ./configs/carsdpm_downstream.json \
  --physics-pretrained ./save_pretrain/20260829_191050/weight/best.pth \
  --save-root ./save

# train with random init
python train.py \
  --config-path ./configs/carsdpm_downstream.json \
  --save-root ./save

# evaluation
python evaluate.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/carsdpm/weight/best.pth \
    --split test \
    --save-pred \
    --save-gt \
    --max-save 8000

# carsirt4 zero-shot
python evaluate.py \
    --config-path ./configs/carsirt4_finetuning.json \
    --weight-path ./save/carsdpm/weight/best.pth \
    --split test \
    --save-root ./save_eval \
    --run-name carsirt4_zero_shot

# carsirt4 fine-tuning
python train.py \
  --config-path ./configs/carsirt4_finetuning.json \
  --carsdpm-pretrained ./save/carsdpm/weight/best.pth \
  --eval-split val \
  --save-root ./save \
  --run-name carsirt4_finetuning

python evaluate.py \
  --config-path ./configs/carsirt4_finetuning.json \
  --weight-path ./save/carsirt4_finetuning/weight/best.pth \
  --split test \
  --save-root ./save_eval \
  --run-name carsirt4_finetuned


# runtime & flops
python benchmark_radiohrformer_runtime.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/dpm/weight/best.pth \
    --device cuda:0 \
    --resolutions 256 512 768 1024 \
    --batch-size 1 \
    --warmup 20 \
    --repeats 100 \
    --precision fp32 \
    --cooldown-seconds 2 \
    --output-dir ./runs/radiohrformer_resolution_runtime


python benchmark_radiohrformer_flops.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/20260814_222709/weight/best.pth \
    --device cuda:0 \
    --resolutions 256 512 768 1024 \
    --batch-size 1 \
    --precision fp32 \
    --output-dir ./runs/radiohrformer_resolution_flops


python convert_existing_predictions_to_rgb.py \
    --pred-dir ./baselines/RadioDiff-k/runs/radiodiffk2_dpm/pred/test/npy \
    --out-dir ./baselines/RadioDiff-k/runs/radiodiffk2_dpm/pred/test/rgb \
    --data-root /home/ailab/Desktop/data/radiomapseer \
    --dataset dpm