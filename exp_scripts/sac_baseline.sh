#!/bin/bash
n_repeats=1

env_names=("half_cheetah")
gpu_ids=(0)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-sac"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --policy_type tanh_gaussian --batch_size 256
          
          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

env_names=("ant")
gpu_ids=(1)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-sac"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --policy_type tanh_gaussian --batch_size 256

          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

env_names=("walker")
gpu_ids=(2)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-sac"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --policy_type tanh_gaussian --batch_size 256

          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

# ### HAND

env_names=("door-zero-one-v0")
gpu_ids=(3)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-sac"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --policy_type rail_gaussian --batch_size 1024
          
          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done

env_names=("pen-zero-one-v0")
gpu_ids=(4)
for gpu_id in "${gpu_ids[@]}"
  do
    for env_name in "${env_names[@]}"
      do
        label="${env_name}-sac"
        for i in $(seq 1 $n_repeats)
        do 
          echo "Starting repeat $i of $label"

          bash exp_scripts/run-gpy.sh $gpu_id python nppac/nppac.py --name $env_name --label $label --use_gpu \
          --policy_type rail_gaussian --batch_size 1024

          echo "Sleeping 2 seconds"
          sleep 2
          echo
        done
      done
  done
