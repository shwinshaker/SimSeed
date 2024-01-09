# SimSeed
A simple yet effective method for seed word-based weakly supervised text classification

## Installation
1. Install anaconda
2. `bash install.sh`
3. Download datasets

## Run

* Random deletion:

Set `pseudo_weak_sup_select_train_aug: randremove` in `config.yaml`, then `./launch.sh`

* Seed deletion:

Set `pseudo_weak_sup_select_train_aug: adremove` in `config.yaml`, then `./launch.sh`

## TODOs
* More cleaning
