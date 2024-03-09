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

cd data
# - 20news-fine
gdown https://drive.google.com/uc?id=1Aysr7udzK2gFVis470iuK3rBws_EHrHB
unzip 20news-fine.zip

# - 20news-coarse
gdown https://drive.google.com/uc?id=1vOBvIWIZhnlOUQnrmzQnwVMfE1nrvhXA
unzip 20news-coarse.zip

# - agnews
gdown https://drive.google.com/uc?id=1_3sgSnnniExqpY-X_mBPGjDCaOnt1WI9
unzip agnews

# - nyt-coarse
gdown https://drive.google.com/uc?id=1E1j8-ya52loWsh9SBeJRxw7SzFPQwiE_
unzip nyt-coarse

# - nyt-fine
gdown https://drive.google.com/uc?id=1Ne9VjoF709WkO78v_lH4M2JtdiPufmkq
unzip nyt-fine

# - rotten_tomatoes
gdown https://drive.google.com/uc?id=1dzyvuMIFyVgzrfB6qQVgTvhdmrCkMQrT
unzip rotten_tomatoes

cd ..
