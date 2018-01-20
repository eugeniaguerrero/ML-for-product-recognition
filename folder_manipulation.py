import os
import numpy as np
from common import *
import cv2

def get_folders(directory_):
    items = os.listdir(directory_)
    folder_present = False
    folders = []
    for d in items:
        if os.path.isdir(os.path.join(directory_, d)):
            folders.append(d)
            folder_present = True

    if not folder_present:
        print("Could not find any folders/categories!")
    return sorted(folders)

def get_image_names(directory_):
    files = os.listdir(directory_)
    image_present = False
    images = []
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            images.append(file)
            image_present = True

    if not image_present:
        print("Could not find any Images!")
    return images

def get_image(filepath):
    img = cv2.imread(filepath)
    resized_image = np.expand_dims(cv2.resize(img, (IM_HEIGHT, IM_WIDTH)), axis=0)
    return resized_image

def dstack_folder(directory_):
    image_list = get_image_names(directory_)
    images = []
    if len(image_list) > 0:
        images = get_image(os.path.join(directory_,image_list[0]))
        for image in image_list[1:]:
            new_image = get_image(os.path.join(directory_, image_list[0]))
            images = np.concatenate((new_image,images),axis = 0)
    return images
