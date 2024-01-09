##################################################
# File  Name: install.sh
#     Author: shwin
# Creat Time: Sat 07 Jan 2023 05:24:27 AM UTC
##################################################

#!/bin/bash

set -e

conda create -n bert python=3.9.1
source /home/chengyu/anaconda3/bin/activate bert
pip install huggingface-hub==0.0.12
pip install transformers==4.8
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install pandas
pip install --user -U nltk
pip install tensorflow==2.9.1
pip install bottleneck
pip install sentencepiece
