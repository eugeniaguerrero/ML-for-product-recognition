import os
import random
import sys
import shutil

percent = 0.9
dir = os.getcwd()
source = "\\product-image-dataset"
train = "\\training_data"
test = "\\test_data"

#create the destination folders
os.makedirs(dir+train)
os.makedirs(dir+test)

#loop through
for d in os.listdir(dir+source):
    i=0
    pictures = []
    dir_name = "\\" + d
    os.makedirs(dir + test + dir_name)
    os.makedirs(dir + train + dir_name)

    for file in os.listdir(dir + source + dir_name):
        pictures.append(file)
        i=i+1

    #make random list
    random.shuffle(pictures)

    j=0
    training_set = []
    test_set = []
    while(j < i*percent):
        training_set.append(pictures[j])
        j=j+1

    while(j < i):
        test_set.append(pictures[j])
        j=j+1

    for pic in training_set:
        shutil.copy(dir+source+dir_name+ "\\" + pic, dir + train + dir_name)

    for pic in test_set:
        shutil.copy(dir+source+dir_name+ "\\" + pic, dir + test + dir_name)
