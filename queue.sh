##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash

# ---------------
config=$1 
[[ -z $config ]] && config="config.yaml"

python config.py -c $config --queue
[[ $? -ne 0 ]] && echo 'exit' && exit 2
checkpoint=$(cat tmp/path.tmp)
path="checkpoints_queue/$checkpoint"
echo $path
if [[ -d $path ]]; then
    read -p "Path already exists! Delete[d] or terminate[*]? " ans
    case $ans in
        d ) rm -r $path;;
        * ) exit 1;;
    esac
fi
mkdir -p $path
scp "tmp/para.json" $path

cp $config $path"/config.yaml"
cp main.py $path
cp config.py $path
cp -r src $path
cp -r script $path
subset_path="$(grep '^train_subset_path' $config | awk '{print$2}' | sed -e "s/^'//" -e "s/'$//")"
if [ ! -z $subset_path ]; then
    cp $subset_path $path
fi

echo "[Path]: $path"
echo "$(date) [Path]: $path" >> logs/queue.txt
