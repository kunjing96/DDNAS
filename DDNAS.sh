#!/bin/bash
# bash ./DDNAS.sh cifar10 1 16 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, track_running_stats, and seed"
  exit 1
fi

dataset=$1
track_running_stats=$2
cutout=$3
seed=$4
space=darts

data_path=~/dataset/${dataset}

save_dir=./output/search-cell-${space}/DDNAS-${dataset}-BN${track_running_stats}

DARTS = "
[[(('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_conv_3x3', 2))]] * 6 + 
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + 
[[(('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_conv_3x3', 2))]] * 5 + 
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + 
[[(('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_conv_3x3', 2))]] * 5
"


OMP_NUM_THREADS=4 python main.py \
	--save_dir ${save_dir} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path  config/DDNAS-OPTS-CIFAR.config \
	--model_config config/DDNAS-ARCHS-CIFAR.config \
	--tau_max 10 --tau_min 0.1 --cutout ${cutout} \
    --track_running_stats ${track_running_stats} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed} \
	--init_genos ${DARTS}
