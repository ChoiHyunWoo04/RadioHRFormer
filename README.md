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
    --weight-path ./save/20260831_063949/weight/best.pth \
    --split test \
    --save-pred \
    --save-gt \
    --max-save 8000


# runtime & flops
python benchmark_radiohrformer_runtime.py \
    --config-path ./configs/carsdpm_downstream.json \
    --weight-path ./save/20260814_222709/weight/best.pth \
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
    --pred-dir ./baselines/RadioDiff/runs/radiodiff_dpm/pred/test/npy \
    --out-dir ./baselines/RadioDiff/runs/radiodiff_dpm/pred/test/rgb \
    --data-root /home/ailab/Desktop/data/radiomapseer \
    --dataset carsdpm