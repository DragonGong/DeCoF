#!/bin/bash

run_sh=$0
to_pred_dir=$1
submit_path=$2

abs_run_sh=$(realpath $run_sh)
run_dir=${abs_run_sh%/run.sh}

# 镜像中有3个python, 分别是python3.7, python3.8 和 python3.10
# 如果要使用python3.7, 可以把逻辑直接写在run.py中, 也可以选择执行下面的命令
# python3.7 $run_dir/model.py $to_pred_dir $submit_path

# 如果要使用python3.8, 则可以使用以下命令
# python3.8 $run_dir/model.py $to_pred_dir $submit_path

# 如果要使用python3.10, 则可以使用以下命令
python3.10 $run_dir/model.py $to_pred_dir $submit_path

#!!! 特别注意, 无论使用哪种命令, 请千万不要使用nohup, 因为如果提交任务的是nohup的话, 容器会直接退出!!!
#!!! 且nohup.out文件在容器中是不会生成的!!!