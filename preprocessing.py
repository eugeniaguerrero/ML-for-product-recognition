from rotation_zoom_flip import *
from common import *
from folder_manipulation import dstack_folder
import os
import cv2

def preprocess_images(picture_batch):
        if rotation == turned.ON:
            picture_batch = random_rotation(picture_batch, rotation_degrees, 0, 1, 2)
        if zoom == turned.ON:
            picture_batch = random_zoom(picture_batch, (zoom_horizontal, zoom_vertical), 0, 1, 2)
        if horizontal_flip == turned.ON:
            picture_batch = flip_horizontal(picture_batch)
        if vertical_flip == turned.ON:
            picture_batch = flip_vertical(picture_batch)
        return picture_batch

'''Testing the preprocessing function'''
x = dstack_folder(os.path.join("validation_data","5055540026268","VID36"))
y = x[0]
y = np.expand_dims(y, axis=0)
z = preprocess_images(y)

cv2.imwrite("Original.jpeg", y[0])
cv2.imwrite("Preprocessed.jpeg", z[0])