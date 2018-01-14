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

def get_substring(filename):

    start = filename.find('_') + 1
    end = filename.rfind('_') - 2

    return filename[start:end]

# Set percentage of images for training (rest are test set)
test_pct = 0.1
validate_pct = 0.1
train_pct = 1 - validate_pct - test_pct

source = "product-image-dataset"
train = "training_data"
validate = "validation_data"
test = "test_data"
dir = os.getcwd()

#os.makedirs(os.path.join(dir, test))
#os.makedirs(os.path.join(dir, validate))
#os.makedirs(os.path.join(dir, train))


#Loop subdirectories
for subdir in os.listdir(os.path.join(dir,source)):
    #os.makedirs(os.path.join(dir, test, subdir))
    #os.makedirs(os.path.join(dir, validate, subdir))
    #os.makedirs(os.path.join(dir, train, subdir))

    stamps = {}
    files = {}
    for f in os.listdir(os.path.join(dir, source, subdir)):

        time_stamp = get_substring(f)
        if stamps.haskey(time_stamp):
            group = stamps(time_stamp)
            files[f] = group

        else:
            if stamps == {}:
                new_group = 0
            else:
                new_group = max(stamps.values()) + 1

            stamps[time_stamp] = new_group
            files[f] = new_group

    #Shuffle list to randomise
    n_groups = max(stamps.values)
    groups = list(range(0, n_groups))
    random.shuffle(groups)

    #Split random list into training and test
    path = os.path.join(dir, source, subdir)
    j=0
    while(j < (n_groups*test_pct)):
        for filename in files:
            if files[filename] == groups[j]:
                shutil.copy(os.path.join(path, filename), os.path.join(dir, test, subdir))
        j+=1

    while(j <  n_groups*(test_pct+validate_pct)):
        for filename in files:
            if files[filename] == groups[j]:
                shutil.copy(os.path.join(path, filename), os.path.join(dir, validate, subdir))
        j+=1

    while(j < n_groups):
        for filename in files:
            if files[filename] == groups[j]:
                shutil.copy(os.path.join(path, filename), os.path.join(dir, train, subdir))
        j+=1
