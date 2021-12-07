#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
GPU=$1
name=${USER}_rlkit_GPU_${GPU//,/-}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$GPU" ${cmd} run -d \
    --name $name \
    -v `pwd`:/rlkit \
    -e PYTHONPATH='/rlkit' \
    -e LOCAL_LOG_DIR='/rlkit/data' \
    -t torch-nppac \
    ${@:2}
