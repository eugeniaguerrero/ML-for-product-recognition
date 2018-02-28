import os
from shutil import move

def store_images_in_class_folder(my_folders, working_dir):
    for folder in my_folders:
        for class_folder in os.listdir(folder):
            if os.path.isdir(os.path.join(working_dir, folder, class_folder)):
                class_path = os.path.join(working_dir, folder, class_folder)
                for video_folder in os.listdir(class_path):
                    if os.path.isdir(os.path.join(class_path, video_folder)):
                        video_path = os.path.join(class_path, video_folder)
                        for picture in os.listdir(video_path):
                            move(os.path.join(video_path, picture), os.path.join(class_path, picture))
                        os.rmdir(video_path)


working_dir = os.getcwd()
folder = ['my_test_data','my_training_data', 'my_validation_data']
store_images_in_class_folder(folder, working_dir)