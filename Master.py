from vgg_net import *

nn = NN()
#nn.debug('debug_folder')
nn.train('test','v1',10)
nn.find_incorrect_classifications('test')
