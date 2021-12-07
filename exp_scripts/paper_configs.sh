#!/bin/bash
n_repeats=1
gp_rank=1

alpha_val=0.1
model_file="trained_gps/gp_halfcheetah_rbf_rank1.pt"
data_file="expert_demonstration_data/hc_action_noise_15.npy"
env_names=("half_cheetah")
kernel_type=rbf

gpu_ids=(0)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-nppac-${kernel_type}"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --use_gp --model_file $model_file --data_file $data_file  --policy_type tanh_gaussian --batch_size 256 \
          --gp_rank $gp_rank --kernel_type $kernel_type --pretrain_policy 400 --use_fixed_alpha --alpha $alpha_val
          
          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

alpha_val=0.1
model_file="trained_gps/gp_ant_rbf_rank1.pt"
data_file="expert_demonstration_data/ant_action_noise_15.npy"
env_names=("ant")
kernel_type=rbf

gpu_ids=(1)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-nppac-${kernel_type}"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --use_gp --model_file $model_file --data_file $data_file  --policy_type tanh_gaussian --batch_size 256 \
          --gp_rank $gp_rank --kernel_type $kernel_type --pretrain_policy 400 --use_fixed_alpha --alpha $alpha_val
          
          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

alpha_val=0.2
model_file="trained_gps/gp_walker_rbf_rank1.pt"
data_file="expert_demonstration_data/walker_action_noise_15.npy"
env_names=("walker")
kernel_type=rbf

gpu_ids=(2)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-nppac-${kernel_type}"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --use_gp --model_file $model_file --data_file $data_file  --policy_type tanh_gaussian --batch_size 256 \
          --gp_rank $gp_rank --kernel_type $kernel_type --pretrain_policy 400 --use_fixed_alpha --alpha $alpha_val
          
          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

# ### HAND

alpha_val=0.5
model_file="trained_gps/gp_door_matern12_rank1.pt"
data_file="expert_demonstration_data/door2_sparse.npy"
env_names=("door-zero-one-v0")
kernel_type=matern12

gpu_ids=(3)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-nppac-${kernel_type}"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --use_gp --model_file $model_file --data_file $data_file  --policy_type rail_gaussian --batch_size 1024 \
          --gp_rank $gp_rank --kernel_type $kernel_type --pretrain_policy 400 --use_fixed_alpha --alpha $alpha_val
          
          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

alpha_val=0.1
model_file="trained_gps/gp_pen_matern52_rank1.pt"
data_file="expert_demonstration_data/pen2_sparse.npy"
env_names=("pen-zero-one-v0")
kernel_type=matern52

gpu_ids=(4)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-nppac-${kernel_type}"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --use_gp --model_file $model_file --data_file $data_file  --policy_type rail_gaussian --batch_size 1024 \
          --gp_rank $gp_rank --kernel_type $kernel_type --pretrain_policy 550 --use_fixed_alpha --alpha $alpha_val
          
          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done
