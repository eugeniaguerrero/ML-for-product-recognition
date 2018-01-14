import os
import numpy as np
from common import *

def get_folders(directory_):
    items = os.listdir(directory_)
    folder_present = False
    folders = []
    for d in items:
        if os.path.isdir(os.path.join(directory_, d)):
            folders = folders.append(d)
            folder_present = True

    if not folder_present:
        print("Could not find any folders/categories!")
    return folders

def get_image_names(directory_):
    files = os.listdir()
    image_present = False

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image = file
            image_present = True
            break
    if not image_present:
        print("Could not find any Images!")

def get_image(filepath):
    import cv2
    img = cv2.imread(filepath)
    resized_image = np.expand_dims(cv2.resize(img, (IM_HEIGHT, IM_WIDTH)), axis=0)
    return resized_image