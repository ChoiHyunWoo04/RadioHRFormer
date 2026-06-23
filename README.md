# visualization
python tools/visualize_physics_targets.py \
  --config-path ./configs/hrt.json \
  --data-root /home/ailab/Desktop/data/radiomapseer \
  --input-mode cars \
  --target-type carsDPM \
  --split train \
  --physics-targets grad,lap,singularity,los,obstacle \
  --geo-precompute-root ./data/precomputed_geo \
  --save-dir ./save/visual_physics_targets \
  --num-samples 4 \
  --tx-channel 2


# pretrain
python train_hrformer_physics_pretrain.py \
  --config-path ./configs/hrt.json \
  --data-root /home/ailab/Desktop/data/radiomapseer \
  --input-mode building \
  --target-type DPM \
  --physics-targets grad,lap,singularity,los,obstacle \
  --geo-precompute-root ./data/precomputed_geo \
  --epochs 200 \
  --batch-size 32 \
  --tx-channel 2

python train_hrformer_physics_pretrain.py \
  --config-path ./configs/hrt.json \
  --data-root /home/ailab/Desktop/data/radiomapseer \
  --input-mode cars \
  --target-type carsDPM \
  --physics-targets grad,lap,singularity,los,obstacle \
  --geo-precompute-root ./data/precomputed_geo \
  --epochs 100 \
  --batch-size 32 \
  --tx-channel 2

# train with pretrained weights
python train_hrformer.py \
  --config-path ./configs/hrt.json \
  --data-root /path/to/RadioMapSeer \
  --input-mode building \
  --target-type DPM \
  --physics-pretrained ./save_pretrain/physics_run/weight/best.pth \
  --run-name hrformer_physics_ft \
  --cuda 0

python train_hrformer.py \
  --config-path ./configs/hrt.json \
  --data-root /home/ailab/Desktop/data/radiomapseer/ \
  --input-mode cars \
  --target-type carsDPM \
  --physics-pretrained ./save_pretrain/physics_run/weight/best.pth \
  --run-name hrformer_physics_ft

# train with random init
python train_hrformer.py \
  --config-path ./configs/hrt.json \
  --data-root /path/to/RadioMapSeer \
  --input-mode building \
  --epochs 200 \
  --batch-size 32 \
  --cuda 0

python train_hrformer.py \
  --config-path ./configs/hrt.json \
  --data-root /path/to/RadioMapSeer \
  --input-mode cars \
  --epochs 200 \
  --batch-size 32 \
  --cuda 0

# evaluation
python evaluate_hrformer.py \
  --config-path ./configs/hrt.json \
  --data-root /home/ailab/Desktop/data/radiomapseer/ \
  --weight-path ./save/cars_run/weight/best.pth \
  --input-mode cars \
  --split test