#File containing the main parameters used for the neural network
from enum import Enum

class turned(Enum):
    ON = 0
    OFF = 1

'''Model parameters'''
#Model: vgg_net
IM_HEIGHT = 100
IM_WIDTH = 100

'''
#Model: Inception V3
IM_HEIGHT = 299
IM_WIDTH = 299
'''

NUMBER_CLASSES = 10
NUMBER_EPOCHS = 50

'''Preprocessing parameters'''
rotation = turned.ON
rotation_degrees = 30

zoom = turned.ON
zoom_horizontal = 0.8
zoom_vertical = 0.8

horizontal_flip = turned.ON
vertical_flip = turned.OFF


