CSA_GPU_SERVER_REPO_HOME="/home/swyamsingh/repos/catsetmat"
PAARTH_CSA_SERVER_REPO_HOME="/home2/e1-313-15477/govind/repos/catsetmat"
DATA_NAME="mag_acm"

# CHANGE THIS BASED ON THE SERVER IT IS RUN ON
REPO_HOME=$CSA_GPU_SERVER_REPO_HOME

DATA_PATH=$REPO_HOME/data/raw/$DATA_NAME/
cd $DATA_PATH

ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-3GrdZzhRSh7iwYPpiexYYN7I6tKUDm5" -O id_p_map.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-1ToWfkh50wlkF_0SLEukMQ871mn2Rfc" -O id_k_map.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1-2F38Udu0rz-ooOOayskYVpF4Ce2N163" -O id_a_map.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1H_kxGBdkgLh5C8H5WOXGhphD6ONCWaf9" -O p_a_list_train.txt
ls $DATA_PATH && wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1adjolcW2nYNu95DQ-tRGLkcYjSt8q9s_" -O p_k_list_train.txt

ls $DATA_PATH && cd -

