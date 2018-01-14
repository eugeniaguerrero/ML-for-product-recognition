from vgg_net import *

nn = NN()
#nn.debug('debug_folder')
nn.train(train_directory_='training_data',validation_directory_='validation_data', model_name= 'v1', epochs=10)