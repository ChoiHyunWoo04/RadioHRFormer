# visualization
python tools/visualize_physics_targets.py --config-path ./configs/hrt.json


# precompute
python tools/precompute_obstacle_targets.py \
  --config-path ./configs/hrt.json


# pretrain
python pretrain_hrformer.py \
  --config-path ./configs/hrt.json \
  --save-root ./save_pretrain

# train with pretrained weights
python train_hrformer.py \
  --config-path ./configs/hrt.json \
  --physics-pretrained ./save_pretrain//weight/best.pth \
  --save-root ./save

# train with random init
python train_hrformer.py \
  --config-path ./configs/hrt.json \
  --save-root ./save

# evaluation
python evaluate_hrformer.py \
  --config-path ./configs/hrt.json \
  --weight-path /path/to/best.pth \
  --split test \
  --save-root ./save_eval