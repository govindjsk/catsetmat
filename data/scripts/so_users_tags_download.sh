CSA_GPU_SERVER_REPO_HOME="/home/swyamsingh/repos/catsetmat"
PAARTH_CSA_SERVER_REPO_HOME="/home2/e1-313-15477/govind/repos/catsetmat"
DATA_NAME="so_users_tags"

# CHANGE THIS BASED ON THE SERVER IT IS RUN ON
REPO_HOME=$CSA_GPU_SERVER_REPO_HOME

DATA_PATH=$REPO_HOME/data/raw/$DATA_NAME/
cd $DATA_PATH

ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-cAxeB7hfZsYmwVljElSotzZlRy1Zvwm" -O id_p_map.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-cwQzJZ4LPnRsH3z2pILLtUN9sKswoO8" -O id_k_map.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-nabJPH5afKnM_wetAvjcnqFjGZZuKIb" -O id_a_map.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-WFFWHX3-D7aIC3fwzdqjyFw-wbrTI-Y" -O p_a_list_train.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-P-NvMUJ8QHT-iMvckLLeLGLda4C5Jp6" -O p_k_list_train.txt

ls $DATA_PATH && cd -
