#!/usr/bin/bash
module load tools computerome_utils/2.0
module unload gcc
module load gcc/11.1.0
module load anaconda3/4.4.0
module load minimap2/2.17r941 samtools/1.10

source ~/.bashrc
conda init bash
conda activate vamb_env

path=/home/projects/cpr_10006/people/paupie/Data/data/airways/
nepoch=500

python3 train_semisupervised.py  --path "$path" --nepoch $nepoch --cuda --supervision 0.5
