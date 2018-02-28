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

vgg_ = VGG(False)#,cached_model="MODEL_OUTPUTS/old_models/1_vgg_netnormal_data+preprocessing.hdf5")#,cached_model="MODEL_OUTPUTS/old_models/intermediate.hdf5")#,cached_model="MODEL_OUTPUTS/old_models/4_vgg_net_magical80Arun.hdf5")
vgg_.train(train_directory_='DATA/training_data', validation_directory_='DATA/training_data', model_description= 'New_Test',epochs=NUMBER_EPOCHS)
datagen = ImageDataGenerator()

validate_generator = datagen.flow_from_directory(
            'DATA/training_data',
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical")  # CHANGE THIS!!!


print(vgg_.model.metrics_names)
print(vgg_.model.evaluate_generator(validate_generator))
#
#find_incorrect_classifications("DATA/FD_Feb22/validation_data",vgg_)
#find_incorrect_classifications('DATA/training_data',vgg_)
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

