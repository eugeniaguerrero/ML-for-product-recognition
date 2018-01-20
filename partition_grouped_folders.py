'''
Splits a folder with source\subdirs\images into:
    test_data\subdirs\images
    validation_data\subdirs\images
    training_data\subdirs\images

Images are grouped according to their timestamps e.g.
file name '5000169217429_2017-06-24_13.46.42.629_p_049.jpg'
has time stamp: '2017-06-24_13.46.42.629'

The order of the groups is shuffled and then the groups are copied into
the test, validation and training folders according the the ratios specified.

'''
import os
import random
import sys
import shutil

# get_timestamp returns the time stamp substring from the file name
def get_timestamp(filename):
    start = filename.find('_') + 1
    end = filename.rfind('_') - 2
    return filename[start:end]

# Set percentage of images for testing and validation (remainder is training set)
test_pct = 0.1
validate_pct = 0.1
train_pct = 1 - validate_pct - test_pct

source = "product-image-dataset"
train = "training_data"
validate = "validation_data"
test = "test_data"
dir = os.getcwd()

source_dir = os.path.join(dir, source)
test_dir = os.path.join(dir, test)
validate_dir = os.path.join(dir, validate)
train_dir = os.path.join(dir, train)

# Delete directories if they already exist, and make new ones
for d in [test_dir, validate_dir, train_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# Loop through the subdirectories of the source data (i.e image class folders)
for subdir in os.listdir(source_dir):

    # Skip any items in the source directory that are not directories
    if not os.path.isdir(os.path.join(source_dir, subdir)):
        continue

    # Make the corresponding subdirectories in each of the destination directories
    os.makedirs(os.path.join(test_dir, subdir))
    os.makedirs(os.path.join(validate_dir, subdir))
    os.makedirs(os.path.join(train_dir, subdir))

    # Loop through the image files in a subdirectory
    stamps = {}
    files = {}
    for f in os.listdir(os.path.join(source_dir, subdir)):

        time_stamp = get_timestamp(f)
        # If the time stamp already exists lookup the group number and add it to the files dictionary
        if time_stamp in stamps:
            group = stamps[time_stamp]
            files[f] = group

        # If it does not exist add it to the stamps dictionary with an incremented group number
        # Add the file the the files dictionary with the new group number
        else:
            if stamps == {}:
                new_group = 0
            else:
                new_group = max(stamps.values()) + 1

            stamps[time_stamp] = new_group
            files[f] = new_group

    #Make a list from 0 to n_groups and randomise the order
    n_groups = max(stamps.values())
    groups = list(range(0, n_groups+1))
    random.shuffle(groups)

    #Split random list into test, validation and training directories
    prefix = "VID"
    path = os.path.join(source_dir, subdir)
    required_frames = 4
    j=0
    while(j < (n_groups*test_pct)):
        if list(files.values()).count(groups[j]) == required_frames:
            video_folder = os.path.join(test_dir, subdir, prefix + str(groups[j]))
            os.makedirs(video_folder)
            for file_name in files:
                if files[file_name] == groups[j]:
                    shutil.copy(os.path.join(path, file_name), video_folder)
        j+=1

    while(j <  n_groups*(test_pct+validate_pct)):
        if list(files.values()).count(j) == required_frames:
            video_folder = os.path.join(validate_dir, subdir, prefix + str(j))
            os.makedirs(video_folder)
            for file_name in files:
                if files[file_name] == groups[j]:
                    shutil.copy(os.path.join(path, file_name), video_folder)
        j+=1

    while(j <= n_groups):
        if list(files.values()).count(j) == required_frames:
            video_folder = os.path.join(train_dir, subdir, prefix + str(j))
            os.makedirs(video_folder)
            for file_name in files:
                if files[file_name] == groups[j]:
                    shutil.copy(os.path.join(path, file_name), video_folder)
        j+=1
