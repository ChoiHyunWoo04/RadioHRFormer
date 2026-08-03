# visualization
python tools/visualize_physics_targets.py --config-path ./configs/carsdpm_pretrain.json


# precompute
python tools/precompute_obstacle_targets.py \
  --config-path ./configs/carsdpm_pretrain.json


# pretrain
python pretrain_hrformer.py \
  --config-path ./configs/carsdpm_pretrain.json \
  --save-root ./save_pretrain

# train with pretrained weights
python train_hrformer.py \
  --config-path ./configs/carsdpm_downstream.json \
  --physics-pretrained ./save_pretrain/20260716_190305_/weight/best.pth \
  --save-root ./save

# train with random init
python train_hrformer.py \
  --config-path ./configs/carsdpm_downstream.json \
  --save-root ./save

# evaluation
python evaluate_hrformer.py \
  --config-path ./configs/carsdpm_downstream.json \
  --weight-path ./save/20260729_212856/weight/best.pth \
  --split test \
  --save-root ./save_eval