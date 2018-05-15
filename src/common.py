##File containing the main parameters used for the neural network


import os

#GANN MUST BE DIVISIBLE BY 4!!!
IM_HEIGHT = 200
IM_WIDTH = 200
RAW_HEIGHT = 1024
RAW_WIDTH = 1280

'''
IM_HEIGHT = 299
IM_WIDTH = 299'''

PRETRAINED_MODEL = False
NUMBER_CLASSES = 20
BATCH_SIZE = 16
NUMBER_EPOCHS = 500
NUMBER_CHANNELS = 3
IMAGES_PER_FOLDER = 12
SEND_TO_SLACK = True

CHECKPOINTS_FOLDER = os.path.join('MODEL_OUTPUTS','checkpoints')
MODEL_SAVE_FOLDER = os.path.join('MODEL_OUTPUTS','models')
OLD_MODELS_FOLDER = os.path.join('MODEL_OUTPUTS','old_models')
TENSORBOARD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','logs')
TENSORBOARD_OLD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','old_logs')
INTERMEDIATE_FILE = os.path.join('MODEL_OUTPUTS','checkpoints','intermediate.hdf5')
JSON_LOG_FILE = os.path.join('MODEL_OUTPUTS','loss_log.json')
JSON_OLD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','old_json')

#THESE PARAMETERS BELOW ARE NOT USED IN MASTER SUBMITTED AS IT USES VGG_TESTING

SOURCE = os.path.join("DATA","product-image-dataset3")
TRAIN_DATA = os.path.join("DATA","f_d22_training_data")#"March-18","training_data")
VALIDATE_DATA = os.path.join("DATA","f_d22_validation_data")#"March-18","validation_data")
TEST_DATA = os.path.join("DATA","March-18","test_data")
DEBUG_FOLDER = os.path.join("DATA","DEBUGGING_DATA","debug_folder")


TRAIN_DATA_GROUPED = os.path.join("DATA","training_data_grouped")
VALIDATE_DATA_GROUPED = os.path.join("DATA","validation_data_grouped")
TEST_DATA_GROUPED = os.path.join("DATA","test_data_grouped")
