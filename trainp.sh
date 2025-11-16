
CUDA_VISIBLE_DEVICES=3,4 accelerate launch --num_processes 2 --main_process_port 58220  train_small.py --config_path configs/trainv1_small_p.yaml --sec_type only_patch

python train_d.py --gpus 3 4


#CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --num_processes 4 --main_process_port 58110  train.py --config configs/trainv1_1.yaml