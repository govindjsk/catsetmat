DATANAME='sample_mag_acm'
NUMSPLITS=5
STARTSPLIT=0
NEGFACTOR=5
TESTRATIO=0.2

python src/train_test_sampler.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT --neg_factor $NEGFACTOR --test_ratio $TESTRATIO
python src/embedding_storer.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT
python src/main.py --data_name $DATANAME --num_splits $NUMSPLITS --start_split $STARTSPLIT