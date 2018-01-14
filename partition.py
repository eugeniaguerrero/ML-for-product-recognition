'''
Splits a folder with source\subdirs\images into:
    training_data\subdirs\images
    test_data\subdirs\images

'percent' is the proportion of randomly selected images for the training set.
'''
import os
import random
import sys
import shutil

# Set percentage of images for training (rest are test set)
percent = 0.9

source = "\\product-image-dataset"
train = "\\training_data"
test = "\\test_data"
dir = os.getcwd()

os.makedirs(dir+train)
os.makedirs(dir+test)

#Loop subdirectories
for sub in os.listdir(dir+source):
    i=0
    pictures = []
    subdir= "\\" + sub
    os.makedirs(dir + test + subdir)
    os.makedirs(dir + train + subdir)

    #Count and create list of files in subdir
    for file in os.listdir(dir + source + subdir):
        pictures.append(file)
        i=i+1

    #Shuffle list to randomise
    random.shuffle(pictures)

    #Split random list into training and test
    j=0
    training_set = []
    test_set = []
    while(j < i*percent):
        training_set.append(pictures[j])
        j= j+1

    while(j < i):
        test_set.append(pictures[j])
        j=j+1

    #Copy to destination folders
    path = dir + source + subdir + "\\"
    for pic in training_set:
        shutil.copy(path + pic, dir + train + subdir)

    for pic in test_set:
        shutil.copy(path + pic, dir + test + subdir)
