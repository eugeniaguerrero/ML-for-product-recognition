from keras.applications.inception_v3 import InceptionV3
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from src.callbacks import *
from src.DATA_PREPARATION.folder_manipulation import *
from src.NN_MODELS.common_network_operations import *
from src.DATA_PREPARATION.data_generator import *
from src.PREPROCESSING.rotation_zoom_flip import *

from PIL import Image

direc = os.path.join('DATA','training_data')
params = {'dir': direc,'batch_size': 8, 'shuffle': True, 'sequence_length' : 4}


validation_generator = DataGenerator(**params, time_distributed=True)
gen = validation_generator.generate()

print("START")
output_new = gen.__next__()

out = 127.5*(output_new[0] + 127.5)
output_new = out.astype('uint8')

for j in range(output_new.shape[0]):
    img = Image.fromarray(output_new[j][0], 'RGB')
    img.save("im{}.png".format(j))
    img.show()



# output_new = gen.__next__()
# print("HERE!")
# output_new = output_new[0].astype('uint8')
# for i in range(output_new.shape[0]):
#     for j in range(output_new.shape[1]):
#         img = Image.fromarray(output_new[i,j], 'RGB')
#         img.save("zim{}T{}.png".format(i,j))
#
#



print("Done")