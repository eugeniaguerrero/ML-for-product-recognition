#File containing the main parameters used for the neural network
import os
#vgg_net images MUST BE DIVISIBLE BY 4!!!
IM_HEIGHT = 100
IM_WIDTH = 100
NUMBER_CHANNELS = 3
FILES_IN_FOLDER = 4
'''
IM_HEIGHT = 299
IM_WIDTH = 299'''
NUMBER_CLASSES = 10
NUMBER_EPOCHS = 3

SEND_TO_SLACK = False

CHECKPOINTS_FOLDER = os.path.join('MODEL_OUTPUTS','checkpoints')
MODEL_SAVE_FOLDER = os.path.join('MODEL_OUTPUTS','models')
OLD_MODELS_FOLDER = os.path.join('MODEL_OUTPUTS','old_models')
TENSORBOARD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','logs')
TENSORBOARD_OLD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','old_logs')
INTERMEDIATE_FILE = os.path.join('MODEL_OUTPUTS','checkpoints','intermediate.hdf5')
JSON_LOG_FILE = os.path.join('MODEL_OUTPUTS','loss_log.json')
JSON_OLD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','old_json')

SOURCE = os.path.join("DATA","product-image-dataset3")
TRAIN_DATA = os.path.join("DATA","training_data")
VALIDATE_DATA = os.path.join("DATA","validation_data")
TEST_DATA = os.path.join("DATA","test_data")

TRAIN_DATA_GROUPED = os.path.join("DATA","training_data_grouped")
VALIDATE_DATA_GROUPED = os.path.join("DATA","validation_data_grouped")
TEST_DATA_GROUPED = os.path.join("DATA","test_data_grouped")