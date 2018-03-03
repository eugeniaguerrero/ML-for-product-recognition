from src.NN_MODELS.cnn_lstm import *
from src.NN_MODELS.inception_v3 import *
from src.NN_MODELS.vgg_testing import *
#from src.NN_MODELS. import *
from src.common import *
#from src.NN_MODELS.network_tests import *
from src.DATA_PREPARATION.partition_grouped import *
from src.DATA_PREPARATION.partition_grouped_folders import *
import keras

#TEST DATA GROUPED
#sort_data(SOURCE,TRAIN_DATA,TEST_DATA,VALIDATE_DATA)
#sort_data_folders(SOURCE,TRAIN_DATA_GROUPED,TEST_DATA_GROUPED,VALIDATE_DATA_GROUPED)

'''
#CHANGED FOR TESTING FOLDERS ETC
cnn_lstm_ = CNN_LSTM()
cnn_lstm_.train(train_directory_='DATA/training_data', validation_directory_='DATA/training_data_grouped',model_description= '', epochs=NUMBER_EPOCHS)
'''

#datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,rotation_range=20)

'''rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,'''

'''
width_shift_range=0.1,
height_shift_range=0.1,
channel_shift_range=0.2,
shear_range=0.35,
zoom_range = [0.7, 1.3])'''

'''
datagenval = ImageDataGenerator(
    rescale=1. / 255)

for image_size in [300]:
    IM_HEIGHT = image_size
    IM_WIDTH = image_size
    vgg_ = VGG(False)
    vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= "image_test_size" + str(image_size), epochs=NUMBER_EPOCHS,datagen=datagen,datagenval=datagenval)
'''

'''
inception_v3_ = INCEPTION_V3()
inception_v3_.train(train_directory_='DATA/training_data', validation_directory_='DATA/training_data', model_description= '', epochs=NUMBER_EPOCHS)
'''

