# visualization
python visualize_physics_targets.py --config-path ./configs/hrt.json


# precompute
python tools/precompute_obstacle_targets.py \
  --config-path ./configs/hrt.json \
  --data-root /home/ailab/Desktop/data/radiomapseer \
  --save-root ./data/precomputed_obstacle \
  --input-mode cars \
  --target-type carsDPM \
  --splits train,val,test \
  --tx-channel 2 \
  --obstacle-channels 0,1 \
  --dtype float16


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