CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --num_processes 4 --main_process_port 58110  train.py --config configs/trainv1_1.yaml
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --num_processes 4 --main_process_port 58110  train.py


python train_d.py --gpus 1 2 3 4