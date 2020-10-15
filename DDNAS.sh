#!/bin/bash
# bash ./DDNAS.sh cifar10 16 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, cutout, and seed"
  exit 1
fi

dataset=$1
cutout=$2
seed=$3
space=darts

data_path=~/dataset/${dataset}

save_dir=./output/search-cell-${space}/DDNAS-${dataset}-BN${track_running_stats}

OMP_NUM_THREADS=4 python main.py \
	--save_dir ${save_dir} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path  config/DDNAS-OPTS-CIFAR.config \
	--model_config config/DDNAS-ARCHS-CIFAR.config \
	--tau_max 10 --tau_min 0.1 --cutout ${cutout} \
	--disturb_rate_max 0.125 --disturb_rate_min 0.0 \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed} \
	--init_genos GDAS
