from folder_manipulation import dstack_folder
from common import *
import os
import numpy as np

folder = os.path.join('training_data','3073781011456','VID80')
img_set = dstack_folder(folder)

# Generate noise with normal distribution
(mean, sd) = (0, 5)
sequence_length = 4
noise = np.random.normal(mean, sd, size = (sequence_length, IM_HEIGHT, IM_WIDTH, 3))

# Add noise
img_set = np.add(img_set,noise)
img_set = img_set.clip(min=0.0)
img_set = img_set.astype('uint8')


#Use to view image
#from PIL import Image
#img = Image.fromarray(formatted[0], 'RGB')
#img.show()

