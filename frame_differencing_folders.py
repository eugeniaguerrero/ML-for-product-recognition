from folder_manipulation import *
import shutil
import cv2
import numpy as np
import os
from skimage.measure import compare_ssim
from scipy import stats


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def mse(imageA, imageB):
    # sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    return err



def increase_brightness(img, value=30):
    img_out = np.copy(img)
    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def frame_difference_collection(source, destination):
    pictures = []
    i = 0

    for filename in sorted(os.listdir(source)):
        print(filename)
        i += 1
        filepath = os.path.join(source, filename)
        image = cv2.imread(filepath)
        pictures.append(image)

    region_size = 10
    im_h = 1024
    im_w = 1280
    bl_h, bl_w = region_size, region_size

    # 1 means the pics are identical
    ssim_threshold = 0.94

    # 0 means the images are identical
    mse_threshold = 18

    index = 0
    for pic in pictures:
        block_img = np.zeros((1024, 1280, 3))
        if index == 3:
            individual_product = pictures[index]
            next_product = pictures[index-1]
        else:
            individual_product = pictures[index]
            next_product = pictures[index+1]

        for row in np.arange(im_h - bl_h + 1, step=bl_h):
            for col in np.arange(im_w - bl_w + 1, step=bl_w):
                grayA = cv2.cvtColor(individual_product[row:row+bl_h, col:col+bl_w], cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(next_product[row:row + bl_h, col:col + bl_w], cv2.COLOR_BGR2GRAY)

                (ssim_score, diff) = compare_ssim(grayA, grayB, full=True)
                mse_score = mse(grayA,grayB)

                # checking to see if the pictures are different
                if (ssim_score < ssim_threshold) or (mse_score > mse_threshold):
                    block_img[row:row + bl_h, col:col + bl_w] = individual_product[row:row + bl_h, col:col + bl_w]

        # new_block_img = block_img
        # for row in np.arange(bl_h, im_h - bl_h + 1, step=bl_h):
        #     for col in np.arange(bl_w, im_w - bl_w + 1, step=bl_w):
        #         mode_value = stats.mode(block_img[row - bl_h:row + 2*bl_h, col - bl_w:col + 2*bl_w])
        #         if mode_value[0][0][0][0] != 0 and mode_value[0][0][0][1] != 0 and mode_value[0][0][0][2] != 0:
        #             new_block_img[row:row + bl_h, col:col + bl_w] = 0
        #             #print(block_img[row - bl_h:row + 2 * bl_h, col - bl_w:col + 2 * bl_w])



        file_name = str(index)+ '_' + filename
        original_file_name = str(index) + '_original_' +filename
        block_img = increase_brightness(block_img, 60)
        individual_product = increase_brightness(individual_product, 60)
        cv2.imwrite(os.path.join(destination, file_name), block_img)
        cv2.imwrite(os.path.join(destination, original_file_name), individual_product)
        index += 1


# The directory where the augmented images are saved
difference = 'frame_difference_sample_data'

# The directory where the "to-be-editted" images are saved
topdir = 'sample_data'

dir = os.getcwd()

if os.path.exists(difference):
    shutil.rmtree(difference)
os.makedirs(difference)

top_folders = get_folders(topdir)

for folder in top_folders:
    class_folder = os.path.join(dir,topdir, folder)
    diff_class_folder = os.path.join(dir, difference, folder)
    os.makedirs(diff_class_folder)
    product_video = get_folders(class_folder)

    for video in product_video:
        video_folder = os.path.join(class_folder, video)
        diff_video_folder = os.path.join(diff_class_folder, video)
        os.makedirs(diff_video_folder)
        image_set = get_image_names(video_folder)
        frame_difference_collection(video_folder, diff_video_folder)
        # print(diff_class_folder)


