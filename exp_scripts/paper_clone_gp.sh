#!/bin/bash

gpu_id=0
env_name=half_cheetah
data_set='expert_demonstration_data/hc_action_noise_15.npy'
kt=rbf
bash exp_scripts/run-gpy.sh $gpu_id python nppac/clone_from_dataset_gp.py --name $env_name --save_policies --data_set $data_set \
    --use_gpu --use_ard --gp_rank 1 --kernel_type $kt --save_dir $env_name

gpu_id=1
env_name=ant
data_set='expert_demonstration_data/ant_action_noise_15.npy'
kt=rbf
bash exp_scripts/run-gpy.sh $gpu_id python nppac/clone_from_dataset_gp.py --name $env_name --save_policies --data_set $data_set \
    --use_gpu --use_ard --gp_rank 1 --kernel_type $kt --save_dir $env_name

gpu_id=2
env_name=walker
data_set='expert_demonstration_data/walker_action_noise_15.npy'
kt=rbf
bash exp_scripts/run-gpy.sh $gpu_id python nppac/clone_from_dataset_gp.py --name $env_name --save_policies --data_set $data_set \
    --use_gpu --use_ard --gp_rank 1 --kernel_type $kt --save_dir $env_name

gpu_id=3
env_name=pen
data_set='expert_demonstration_data/pen2_sparse.npy'
kt=matern52
bash exp_scripts/run-gpy.sh $gpu_id python nppac/clone_from_dataset_gp.py --name $env_name --save_policies --data_set $data_set \
    --use_gpu --use_ard --gp_rank 1 --kernel_type $kt --save_dir $env_name

gpu_id=4
env_name=door
data_set='expert_demonstration_data/door2_sparse.npy'
kt=matern12
bash exp_scripts/run-gpy.sh $gpu_id python nppac/clone_from_dataset_gp.py --name $env_name --save_policies --data_set $data_set \
    --use_gpu --use_ard --gp_rank 1 --kernel_type $kt --save_dir $env_name
