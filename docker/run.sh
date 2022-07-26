#!/bin/sh
echo "useage: $_ <GPUS-for-docker> script.py <arg1> <arg2>"
docker run --gpus $1 -it --network host -w /workspace/code --env "PYTHONPATH=/workspace/code" --rm -v $PWD:/workspace/code -v /raid/users/$USER:/workspace/data gwtdocker:latest python3 $2 $3 $4
