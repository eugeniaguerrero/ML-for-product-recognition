from src.NN_MODELS.cnn_lstm import *
from src.NN_MODELS.inception_v3 import *
from src.NN_MODELS.vgg_net import *
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
cnn_lstm_.train(train_directory_='DATA/training_data_grouped', validation_directory_='DATA/training_data_grouped',model_description= '', epochs=NUMBER_EPOCHS)
'''


vgg_ = VGG()
vgg_.train(train_directory_=DEBUG_FOLDER, validation_directory_=DEBUG_FOLDER, model_description= 'normal_data+preprocessing', epochs=NUMBER_EPOCHS)
'''
#TEST 1 - vgg normal data
TRAIN_DATA = os.path.join("DATA","training_data")
VALIDATE_DATA = os.path.join("DATA","training_data")
TEST_DATA = os.path.join("DATA","test_data")

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

vgg_ = VGG()
vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= 'normal_data', epochs=NUMBER_EPOCHS,datagen=datagen)


#TEST 2 - vgg frame_diff data
FTRAIN_DATA = os.path.join("DATA","training_data")
FVALIDATE_DATA = os.path.join("DATA","training_data")
TEST_DATA = os.path.join("DATA","test_data")

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

vgg_ = VGG()
vgg_.train(train_directory_=FTRAIN_DATA, validation_directory_=FVALIDATE_DATA, model_description= 'frame_diff', epochs=NUMBER_EPOCHS,datagen=datagen)


#TEST 3 - vgg normal data + rot
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.35,
    zoom_range=[0.7,1.3],
    channel_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)
vgg_ = VGG()
vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= 'normal_data+preprocessing', epochs=NUMBER_EPOCHS,datagen=datagen)


#TEST 4 - vgg frame_diff data + rot
keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.35,
    zoom_range=[0.7,1.3],
    channel_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

vgg_ = VGG()
vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= 'normal_data+preprocessing', epochs=NUMBER_EPOCHS,datagen=datagen)
'''
'''
inception_v3_ = INCEPTION_V3()
inception_v3_.train(train_directory_='DATA/training_data', validation_directory_='DATA/training_data', model_description= '', epochs=NUMBER_EPOCHS)
'''

