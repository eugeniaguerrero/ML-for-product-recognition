from src.NN_MODELS.cnn_lstm import *
from src.NN_MODELS.inception_v3 import *
from src.NN_MODELS.vgg_net_our_datagen import *
from src.NN_MODELS.gann import *
#from src.NN_MODELS. import *
from src.common import *
from src.DATA_PREPARATION.partition_grouped import *
from src.DATA_PREPARATION.partition_grouped_folders import *

from src.DATA_PREPARATION.folder_manipulation import *

#TEST DATA GROUPED
#sort_data(SOURCE,TRAIN_DATA,TEST_DATA,VALIDATE_DATA)
#sort_data_folders(SOURCE,TRAIN_DATA_GROUPED,TEST_DATA_GROUPED,VALIDATE_DATA_GROUPED)

NUMBER_CLASSES = get_folders(TRAIN_DATA)


wgann_ = WGANN()
wgann_.train()


'''
#CHANGED FOR TESTING FOLDERS ETC
cnn_lstm_ = CNN_LSTM()
cnn_lstm_.train(train_directory_='DATA/training_data_grouped', validation_directory_='DATA/training_data_grouped',model_description= '', epochs=NUMBER_EPOCHS)
'''

#vgg_ = VGG()
#vgg_.train(train_directory_='DATA/training_data', validation_directory_='DATA/training_data', model_description= '', epochs=NUMBER_EPOCHS)

'''
inception_v3_ = INCEPTION_V3()
inception_v3_.train(train_directory_='DATA/training_data', validation_directory_='DATA/training_data', model_description= '', epochs=NUMBER_EPOCHS)
'''

