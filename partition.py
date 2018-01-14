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
test_pct = 0.1
validate_pct = 0.1
train_pct = 1 - validate_pct - test_pct

source = "\\product-image-dataset"
train = "\\training_data"
validate = "\\validation_data"
test = "\\test_data"
dir = os.getcwd()

os.makedirs(dir + test)
os.makedirs(dir + validate)
os.makedirs(dir + train)

#Loop subdirectories
n_test = 0
n_validate = 0
n_train = 0
total = 0
for sub in os.listdir(dir+source):

    subdir= "\\" + sub
    os.makedirs(dir + test + subdir)
    os.makedirs(dir + validate + subdir)
    os.makedirs(dir + train + subdir)

    #Count and create list of files in subdir
    pictures = []
    i = 0
    for file in os.listdir(dir + source + subdir):
        pictures.append(file)
        i=i+1

    #Shuffle list to randomise
    random.shuffle(pictures)

    #Split random list into training and test
    j=0

    test_set = []
    while (j < i * test_pct):
        test_set.append(pictures[j])
        j = j + 1

    validate_set = []
    while(j < i*(test_pct+validate_pct)):
        validate_set.append(pictures[j])
        j = j+1

    training_set = []
    while(j < i):
        training_set.append(pictures[j])
        j = j+1

    #Copy to destination folders
    path = dir + source + subdir + "\\"


    for pic in test_set:
        shutil.copy(path + pic, dir + test + subdir)
        n_test = n_test+1


    for pic in validate_set:
        shutil.copy(path + pic, dir + validate + subdir)
        n_validate = n_validate + 1


    for pic in training_set:
        shutil.copy(path + pic, dir + train + subdir)
        n_train = n_train + 1

    #update running total
    total = total + i

print("There are {} test data images".format(n_test))
print("There are {} validation data images".format(n_validate))
print("There are {} training images".format(n_train))
print("There were {} images in the original data".format(total))


