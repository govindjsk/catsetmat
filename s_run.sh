#!/bin/sh

#SBATCH --job-name=csm
#SBATCH --ntasks=3
#SBATCH --time=22:00:00
#SBATCH --output=csm%j.out
#SBATCH --partition=cl1_all_64C

DATANAME="sample_mag_acm"
NUMSPLITS=5
STARTSPLIT=0
NEGFACTOR=5
TESTRATIO=0.2
MODELNAME='catsetmat'
DIM=16
LR=0.001

python src/train_test_sampler.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --neg_factor $NEGFACTOR --test_ratio $TESTRATIO
python src/embedding_storer.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --dim $DIM
python src/main.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --model_name $MODELNAME --dim $DIM --lr $LR

MODELNAME='fspool'
python src/main.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --model_name $MODELNAME --dim $DIM --lr $LR

#MODELNAME='n2v'
python src/main.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --model_name $MODELNAME --dim $DIM --lr $LR
