from folder_manipulation import *
import shutil
import cv2
import numpy as np
import os
from skimage.measure import compare_ssim
from histogram_equalisation import *
import time



# need to remove .DS files
def mse(imageA, imageB):
    # sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    return err



def increase_brightness(img, value):
    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img



def convert_to_grey_image(image):
    bilateral_filtered_image = cv2.bilateralFilter(image, 7, 150, 150)
    gray_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    return gray_image


def crop_image(img, individual_product):

    background_pixel = 255

    for row in np.arange(img.shape[0] - 1):
        if all(x == background_pixel for x in img[row, :]):
            continue
        else:
            break
    x_bottom = row

    for row in np.arange(img.shape[0]-1, 0, -1):
        if all(x == background_pixel for x in img[row, :]):
            continue
        else:
            break
    x_top = row

    for col in np.arange(img.shape[1] - 1):
        if all(x == background_pixel for x in img[ :,col]):
            continue
        else:
            break
    y_bottom = col

    for col in np.arange(img.shape[1]-1, 0, -1):
        if all(x == background_pixel for x in img[ : , col]):
            continue
        else:
            break
    y_top = col

    cropped_image = individual_product[x_bottom: x_top, y_bottom: y_top, : ]

    return cropped_image



def kernel_compare(individual_product, next_product):
    region_size = 25
    im_h = 1024
    im_w = 1280
    bl_h, bl_w = region_size, region_size

    # 1 means the pics are identical. Try 85
    ssim_threshold = 0.85
    # 0 means the images are identical. Try 25
    mse_threshold = 30

    gray_img = np.zeros((1024, 1280))
    # include this if we want to start with a white square
    gray_img.fill(255)

    grayA = cv2.cvtColor(individual_product, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(next_product, cv2.COLOR_BGR2GRAY)

    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            grayA_kernel = grayA[row:row + bl_h, col:col + bl_w]
            grayB_kernel = grayB[row:row + bl_h, col:col + bl_w]

            (ssim_score, diff) = compare_ssim(grayA_kernel, grayB_kernel, full=True)
            mse_score = mse(grayA_kernel, grayB_kernel)

            # checking to see if the pictures are different
            if (ssim_score < ssim_threshold) and (mse_score > mse_threshold):
                gray_img[row:row + bl_h, col:col + bl_w] = grayA[row:row + bl_h, col:col + bl_w]


    gray_img = gray_img.astype(np.uint8)

    thresh = 254.9
    gray_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)[1]


    _ , contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if we have no contours then there is no difference between frames
    if contours == []:
        return individual_product

    largest_area = sorted(contours, key=cv2.contourArea)[-1]


    blank_block_img = np.zeros((1024, 1280, 3))
    blank_block_img.fill(255)
    blank_block_img = np.uint8(blank_block_img)

    cv2.drawContours(blank_block_img, largest_area, -1, (100, 0, 255), 2)

    grey_block_img = cv2.cvtColor(blank_block_img, cv2.COLOR_BGR2GRAY)
    cropped_block_img = crop_image(grey_block_img, individual_product)

    cropped_block_img = CLAHE_equalisation(cropped_block_img)

    return cropped_block_img


def contor_compare(individual_product, next_product):


    brighter_next_product = increase_brightness(next_product, 30)
    brighter_individual_product = increase_brightness(individual_product, 30)


    process_individual_product = convert_to_grey_image(brighter_individual_product)
    process_next_product = convert_to_grey_image(brighter_next_product)

    # image subtraction
    image_sub = cv2.absdiff(process_individual_product, process_next_product)

    # we threshold the image to make it more prominent
    kernel = np.ones((5, 5), np.uint8)
    close_operated_image = cv2.morphologyEx(image_sub, cv2.MORPH_CLOSE, kernel)
    _, thresholded = cv2.threshold(close_operated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # medianBlur removes noise
    median = cv2.medianBlur(thresholded, 5)

    cropped_median = crop_image(median, individual_product)

    # _, contours, _ = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(brighter_individual_product, contours, -1, (100, 0, 255), 2)

    return cropped_median


def frame_difference_collection(source, destination):
    pictures = []
    i = 0

    for filename in sorted(os.listdir(source)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            print(filename)
            i += 1
            filepath = os.path.join(source, filename)
            image = cv2.imread(filepath)
            pictures.append(image)

    index = 0
    for pic in pictures:
        if index == len(pictures)-1:
            individual_product = pictures[index]
            next_product = pictures[index-1]
        else:
            individual_product = pictures[index]
            next_product = pictures[index+1]

        new_img = kernel_compare(individual_product, next_product)
        # new_img = contor_compare(individual_product, next_product)

        # individual_product = increase_brightness(individual_product, 30)

        file_name = str(index)+ '_' + filename
        original_file_name = str(index) + '_original_' +filename

        cv2.imwrite(os.path.join(destination, file_name), new_img)
        cv2.imwrite(os.path.join(destination, original_file_name), individual_product)

        index += 1


# The directory where the augmented images are going to be saved
difference = 'frame_difference_sample_data'

# The directory where the "to-be-editted" images are already saved

topdir = 'sample_data'

# topdir = 'training_data'

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

