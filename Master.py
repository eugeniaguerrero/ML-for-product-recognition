from src.NN_MODELS.cnn_lstm import *
from src.common import *

nn = NN()
nn.train(train_directory_='training_data', validation_directory_='validation_data', model_name= 'cnn_lstm_50_epochs_v1', epochs=NUMBER_EPOCHS)
#nn.debug('debug_folder')
#nn.find_incorrect_classifications('test')
#nn.clean_up_logs()
