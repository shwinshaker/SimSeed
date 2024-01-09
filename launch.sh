##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash

# ------- env setup -----------
data_dir='/home/chengyu/bert_classification/data'
conda_env='bert'
conda_version='anaconda3'
mkdir -p checkpoints
mkdir -p logs
mkdir -p tmp

config=$1 
[[ -z $config ]] && config="config.yaml"
sed -i "s@^data_dir:.*@data_dir: '$data_dir'@g" $config # use @ as deliminator because slash / conflicts with path

python config.py -c $config
[[ $? -ne 0 ]] && echo 'exit' && exit 2
checkpoint=$(cat tmp/path.tmp)
path="checkpoints/$checkpoint"
echo $path
mkdir -p $path

cp "tmp/para.json" $path
cp $config $path"/config.yaml"
cp main.py $path
cp config.py $path
cp -r src $path
cp -r script $path
subset_path="$(grep '^train_subset_path' $config | awk '{print$2}' | sed -e "s/^'//" -e "s/'$//")"
if [ ! -z $subset_path ]; then
    cp $subset_path $path
fi

trim() {
  local s2 s="$*"
  until s2="${s#[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  until s2="${s%[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  echo "$s"
}
source /home/chengyu/$conda_version/bin/activate $conda_env
cur=$(pwd)
cd $path
if [ $(cat config.yaml | grep "^test:" | awk '{print$2}') == 'True' ]; then
    python main.py | tee train.out
else
    python main.py > train.out 2>&1 &
fi
pid=$!
gpu_id=$(trim $(grep 'gpu_id' config.yaml | awk -F ':' '{print$2}'))
echo "[$pid] [$gpu_id] [Path]: $path"
echo "s [$pid] [$gpu_id] $(date) [Path]: $path" >> $cur/logs/log.txt
cd $cur
