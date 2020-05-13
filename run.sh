#!/bin/sh

DATANAME="sample_mag_acm"
NUMSPLITS=4
STARTSPLIT=0
NEGFACTOR=5
TESTRATIO=0.2
MODELNAME='catsetmat'
DIM=16

python src/train_test_sampler.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --neg_factor $NEGFACTOR --test_ratio $TESTRATIO
python src/embedding_storer.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --dim $DIM
python src/main.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --model_name $MODELNAME --dim $DIM
