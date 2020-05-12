#!/bin/sh

#SBATCH --job-name=csm
#SBATCH --ntasks=3
#SBATCH --time=10:00:00
#SBATCH --output=csm%j.out
#SBATCH --gres=gpu:4
#SBATCH --partition=cl2_all_8G

DATANAME="sample_mag_acm"
NUMSPLITS=4
STARTSPLIT=0
NEGFACTOR=5
TESTRATIO=0.2
MODELNAME='catsetmat'
DIM=16
which python
python src/train_test_sampler.py --help
# /home/swyamsingh/repos/catsetmat/csm_env/bin/python src/train_test_sampler.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --neg_factor $NEGFACTOR --test_ratio $TESTRATIO
# python src/embedding_storer.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --dim $DIM
# python src/main.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --model_name $MODELNAME --dim $DIM
