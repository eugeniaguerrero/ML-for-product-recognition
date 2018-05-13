from src.NN_MODELS.cnn_lstm_ourdatagen import *
#from src.NN_MODELS.inception_v3_our_datagen import *
#from src.NN_MODELS.vgg_testing import *
#from src.NN_MODELS. import *
from src.common import *
#from src.NN_MODELS.network_tests import *
#from src.DATA_PREPARATION.partition_grouped import *
#from src.DATA_PREPARATION.partition_grouped_folders import *
import keras
from bayes_opt import BayesianOptimization
import time

FD_DATA_LOC = os.path.join("DATA","ambient_data","raw_data")#"1104_raw_full_ambient")#,"1104_FD_Full_Set")
NORMAL_DATA_LOC =  os.path.join("DATA","ambient_data","raw_data")


##############################################################################
################# FRAME DIFF VS NON-FRAME DIFF ###########################################
############################################################################

IM_HEIGHT=250
FD = "True"
Preprop = "True"
conv1_size = 6
cached_model= None
moment = 0.9
decay = 6
dense_size = 8
conv2_size = 7
conv1_size = 6
lr=4
NUMBER_EPOCHS = 250
IM_WIDTH=250

FD = "True"
TRAIN_DATA = os.path.join(FD_DATA_LOC,"training_data")
VALIDATE_DATA = os.path.join(FD_DATA_LOC,"validation_data")

#STRING TO SAVE MODEL AS
vals = "LR-" + str(lr) + "_C1-" + str(conv1_size) + "_C2-" + str(conv2_size) + "_DS-" + str(dense_size)
vals +=  "_D-" + str(decay) + "_M-" + str(moment) + "_IM-" + str(IM_WIDTH) + "_FD-" + FD + "_Time_Stamp-" + str(time.time()) + "_Preprop-" + Preprop
cnn = CNN_LSTM()
cnn.train(train_directory_= TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description="CNN", epochs=NUMBER_EPOCHS)
