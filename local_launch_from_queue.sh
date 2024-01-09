##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

# -- config local working dir
server=${1:-'local'}
gpu_id=${2:-'2'}
remote_dir=$(parse_yaml server_setting.yaml | grep $server | grep remote_dir | awk -F '=' '{print$2}' | tr -d '"')
remote_data_dir=$(parse_yaml server_setting.yaml | grep $server | grep remote_data_dir | awk -F '=' '{print$2}' | tr -d '"')
remote_env=$(parse_yaml server_setting.yaml | grep $server | grep remote_env | awk -F '=' '{print$2}' | tr -d '"')
remote_conda_version=$(parse_yaml server_setting.yaml | grep $server | grep remote_conda_version | awk -F '=' '{print$2}' | tr -d '"')
mkdir -p $remote_dir
mkdir -p $remote_dir/checkpoints
mkdir -p $remote_dir/logs
mkdir -p $remote_dir/tmp

# -- get path from queue log
queue_path=$(head -1 logs/queue.txt | awk '{print$NF}')
if [ -z $queue_path ]; then
    echo "Queue is empty. exit!"
    exit 2
fi
checkpoint=$(echo $queue_path | awk -F '/' '{print$2}')
## change gpu id
sed -i "s/^gpu_id: .*$/gpu_id: $gpu_id/g" $queue_path'/config.yaml'
jq '.gpu_id='$gpu_id'' $queue_path'/para.json' > "tmp/para_queue.json" && mv "tmp/para_queue.json" $queue_path'/para.json'
## change data dir
sed -i "s@^data_dir:.*@data_dir: '$remote_data_dir'@g" $queue_path'/config.yaml'
remote_data_dir_parse="\"$(echo $remote_data_dir | sed 's/\//\\\//g')\""
jq -r '.data_dir='$remote_data_dir_parse'' $queue_path'/para.json' > "tmp/para_queue.json" && mv "tmp/para_queue.json" $queue_path'/para.json'

# -- copy queue path to remote
path="$remote_dir/checkpoints/$checkpoint"
echo $path
mkdir -p $path
cp -r $queue_path/* $path

# -- write run file
cat <<EOT > tmp/run.sh
trim() {
  local s2 s="\$*"
  until s2="\${s#[[:space:]]}"; [ "\$s2" = "\$s" ]; do s="\$s2"; done
  until s2="\${s%[[:space:]]}"; [ "\$s2" = "\$s" ]; do s="\$s2"; done
  echo "\$s"
}

source /home/chengyu/$remote_conda_version/bin/activate $remote_env
cd $path
if [ \$(cat config.yaml | grep "^test:" | awk '{print\$2}') == 'True' ]; then
    python main.py | tee train.out
else
    python main.py > train.out 2>&1 &
fi
pid=\$!
gpu_id=\$(trim \$(grep 'gpu_id' config.yaml | awk -F ':' '{print\$2}'))
echo "[$server] [\$pid] [\$gpu_id] [Path]: $path"
EOT


# -- remote run
cp tmp/run.sh $path
bash $path/run.sh >> logs/log.txt
cat logs/log.txt | tail -n1

# -- successful run - deque
sed -i 1d logs/queue.txt
rm -r $queue_path