##################################################
# File  Name: install.sh
#     Author: shwin
# Creat Time: Sat 07 Jan 2023 05:24:27 AM UTC
##################################################

#!/bin/bash

set -e

## create env
conda create -n bert2 python=3.9.1
source /home/chengyu/anaconda3/bin/activate bert2
pip install huggingface-hub==0.0.12
pip install transformers==4.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pandas
pip install --user -U nltk
pip install tensorflow==2.9.1
pip install bottleneck
pip install sentencepiece

mkdir -p checkpoints
mkdir -p logs
mkdir -p tmp

## download dataset
mkdir -p data
pip3 install gdown

# - 20news-fine
cd data
gdown https://drive.google.com/uc?id=1Aysr7udzK2gFVis470iuK3rBws_EHrHB
unzip 20news-fine.zip
cd ..
