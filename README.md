# visualization
python tools/visualize_physics_targets.py --config-path ./configs/carsdpm_pretrain.json


# precompute
python tools/precompute_obstacle_targets.py \
  --config-path ./configs/carsdpm_pretrain.json


# pretrain
python pretrain.py \
  --config-path ./configs/carsdpm_pretrain.json \
  --save-root ./save_pretrain

# train with pretrained weights
python train.py \
  --config-path ./configs/carsdpm_downstream.json \
  --physics-pretrained ./save_pretrain/20260811_181055/weight/best.pth \
  --save-root ./save

# train with random init
python train.py \
  --config-path ./configs/carsdpm_downstream.json \
  --save-root ./save

# evaluation
python evaluate.py \
  --config-path ./configs/carsdpm_downstream.json \
  --weight-path ./save/20260811_180902/weight/best.pth \
  --split test \
  --save-root ./save_eval


# runtime & flops
python benchmark_radiohrformer_runtime.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/20260729_212856/weight/best.pth \
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
    --weight-path ./save/20260729_212856/weight/best.pth \
    --device cuda:0 \
    --resolutions 256 512 768 1024 \
    --batch-size 1 \
    --precision fp32 \
    --output-dir ./runs/radiohrformer_resolution_flops