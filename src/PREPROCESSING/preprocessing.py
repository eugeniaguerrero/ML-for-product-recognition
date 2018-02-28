from src.PREPROCESSING.rotation_zoom_flip import *
from src.PREPROCESSING.histogram_equalisation import *
from src.common import *
from src.DATA_PREPARATION.folder_manipulation import dstack_folder
import os
import cv2

class Preprocessing:
    def __init__(self,
                 rotation = False,
                 rotation_degrees = 360,
                 zoom = False,
                 zoom_max = 0.9, # 0.8
                 horizontal_flip = False,
                 vertical_flip = False,
                 histogram_equalisation = False):

        self.rotation_ = rotation
        self.rotation_degrees_ = rotation_degrees
        self.zoom_ = zoom
        self.zoom_max_ = zoom_max
        self.horizontal_flip_ = horizontal_flip
        self.vertical_flip_ = vertical_flip
        self.histogram_equalisation_ = histogram_equalisation

    def preprocess_images(self, picture_batch):

        #resize from 5d to 4d
        preprocessed_images = picture_batch
        if self.histogram_equalisation_:
            preprocessed_images = CLAHE_equalisation(preprocessed_images)

        if self.rotation_ == True:
             preprocessed_images = random_rotation(preprocessed_images, self.rotation_degrees_, 0, 1, 2)
        if self.zoom_ == True:
            preprocessed_images = random_zoom(preprocessed_images, self.zoom_max_, 0, 1, 2)
        if self.horizontal_flip_ == True:
            preprocessed_images = flip_horizontal(preprocessed_images)
        if self.vertical_flip_ == True:
            preprocessed_images = flip_vertical(preprocessed_images)

        return preprocessed_images



#def output_one_image(self):
#    cv2.imwrite("Preprocessed.jpeg", self.picture_batch_[0])

# # #Testing whether preprocessing works
# pictures_batch = dstack_folder(os.path.join("DATA","validation_data","5055540026268","VID26"))
# preprocessing = Preprocessing(pictures_batch, rotation=True, rotation_degrees=30)
# preprocessing.output_one_image()
# cv2.imwrite("Original.jpeg", pictures_batch[0])












































# from src.PREPROCESSING.rotation_zoom_flip import *
# from src.common import *
# from src.DATA_PREPARATION.folder_manipulation import dstack_folder
# import os
# import cv2
#
# class Preprocessing:
#     def __init__(self,
#                  rotation = False,
#                  rotation_degrees = 360,
#                  zoom = False,
#                  zoom_horizontal = 0.8,
#                  zoom_vertical = 0.8,
#                  horizontal_flip = False,
#                  vertical_flip = False):
#
#         #self.picture_batch_ = pictures_batch
#         self.rotation_ = rotation
#         self.rotation_degrees_ = rotation_degrees
#         self.zoom_ = zoom
#         self.zoom_horizontal_ = zoom_horizontal
#         self.zoom_vertical_ = zoom_vertical
#         self.horizontal_flip_ = horizontal_flip
#         self.vertical_flip_ = vertical_flip
#
#     def preprocess_images(self, picture_batch):
#         print("Called")
#
#         #Get the dimensions
#         n_batch = picture_batch.shape[0]
#         sequence_length = picture_batch.shape[1]
#
#         for i in range(n_batch):
#             pic_time = picture_batch[i]
#             if self.rotation_ == True:
#                 preprocessed_images = random_rotation(pic_time, self.rotation_degrees_, 0, 1, 2)
#             if self.zoom_ == True:
#                 preprocessed_images = random_zoom(pic_time, (self.zoom_horizontal_, self.zoom_vertical_), 0, 1, 2)
#             if self.horizontal_flip_ == True:
#                 preprocessed_images = flip_horizontal(pic_time)
#             if self.vertical_flip_ == True:
#                 preprocessed_images = flip_vertical(pic_time)
#             picture_batch[i] = pic_time
#
#         return picture_batch
