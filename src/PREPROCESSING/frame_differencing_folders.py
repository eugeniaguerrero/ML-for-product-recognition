from GroupProject.src.DATA_PREPARATION.folder_manipulation import *
import shutil
import cv2
import numpy as np
import os
from skimage.measure import compare_ssim
from GroupProject.src.PREPROCESSING.histogram_equalisation import *
import time
from skimage.util.shape import view_as_blocks

def mse(imageA, imageB):
    # sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    return err

def square_border(img):
    if  img.shape[0] > img.shape[1]:
        border_size = int((img.shape[0] - img.shape[1])/2)
        bordered_img = cv2.copyMakeBorder(img, top=0, bottom=0, left=border_size, right=border_size,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        border_size = int((img.shape[1] - img.shape[0])/2)
        bordered_img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=0, right=0,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return bordered_img

def kernel_compare(individual_product, next_product):

    region_size = 64
    im_h = 1024
    im_w = 1280
    bl_h, bl_w = region_size, region_size

    # 1 means the pics are identical. Try 85
    ssim_threshold = 0.85
    # 0 means the images are identical. Try 25
    # mse_threshold = 30

    gray_img = np.zeros((im_h, im_w))
    # include this if we want to start with a white square
    gray_img.fill(255)

    grayA = cv2.cvtColor(individual_product, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(next_product, cv2.COLOR_BGR2GRAY)

    # for row in np.arange(im_h - bl_h + 1, step=bl_h):
    #     for col in np.arange(im_w - bl_w + 1, step=bl_w):
    #
    #         grayA_kernel = grayA[row:row + bl_h, col:col + bl_w]
    #         grayB_kernel = grayB[row:row + bl_h, col:col + bl_w]
    #
    #         (ssim_score, diff) = compare_ssim(grayA_kernel, grayB_kernel, full=True)
    #
    #         #mse_score = mse(grayA_kernel, grayB_kernel)
    #
    #         # checking to see if the pictures are different
    #         if (ssim_score < ssim_threshold): #(mse_score > mse_threshold): #
    #             gray_img[row:row + bl_h, col:col + bl_w] = grayA[row:row + bl_h, col:col + bl_w]

    grayA_blocks = view_as_blocks(grayA, (region_size, region_size))
    grayB_blocks = view_as_blocks(grayB, (region_size, region_size))

    grayA_blocks_reshaped = np.reshape(grayA_blocks, (-1, region_size, region_size))
    grayB_blocks_reshaped = np.reshape(grayB_blocks, (-1, region_size, region_size))

    ssim_array = map(lambda v1, v2: compare_ssim(v1, v2), grayA_blocks_reshaped, grayB_blocks_reshaped)

    # ssim_array = np.fromiter(ssim_array, dtype=np.uint8)
    ssim_array = np.array(list(ssim_array))

    ssim_array_reshape = np.reshape(ssim_array, (int(im_h / region_size), int(im_w / region_size)))
    ssim_array_reshape = np.where(ssim_array_reshape < ssim_threshold, 1, 0)
    ssim_array_full_size = np.kron(ssim_array_reshape, np.ones((region_size, region_size)))

    gray_img = np.multiply(ssim_array_full_size, grayA)

    gray_img = gray_img.astype(np.uint8)

    # thresh = 254.99
    # gray_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)[1]

    thresh = 0
    color = 255
    gray_img = cv2.threshold(gray_img, thresh, color, cv2.THRESH_BINARY)[1]

    _, contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if we have no contours then there is no difference between frames
    if contours == []:
        return individual_product

    largest_area = sorted(contours, key=cv2.contourArea)[-1]

    x, y, w, h = cv2.boundingRect(largest_area)

    individual_product = individual_product[y: y + h, x: x + w]

    cropped_block_img = CLAHE_equalisation(individual_product)

    bordered_img = square_border(cropped_block_img)

    return bordered_img


def frame_difference_collection(source, destination):
    i = 0
    num_of_pics = 0
    for filename in sorted(os.listdir(source)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            num_of_pics = num_of_pics + 1

    pictures = np.empty((num_of_pics, 1024, 1280, 3), dtype=np.uint8)
    for filename in sorted(os.listdir(source)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(source, filename)
            image = cv2.imread(filepath)
            pictures[i, ...] = image
            i += 1

    index = 0

    for _ in pictures:
        if index == len(pictures)-1:
            individual_product = pictures[index]
            next_product = pictures[index-1]
        else:
            individual_product = pictures[index]
            next_product = pictures[index+1]

        new_img = kernel_compare(individual_product, next_product)

        file_name = str(index) + '_' + filename
        original_file_name = str(index) + '_original_' + filename

        cv2.imwrite(os.path.join(destination, file_name), new_img)
        cv2.imwrite(os.path.join(destination, original_file_name), individual_product)

        index += 1

# The directory where the augmented images are going to be saved
difference = 'frame_difference_sample_data'

# The directory where the "to-be-edited" images are already saved
# topdir = 'sample_data'

topdir = 'training_data'

dir = os.getcwd()
print(dir)
exit()

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
        start = time.time()
        frame_difference_collection(video_folder, diff_video_folder)
        end = time.time()
        print("Time", end - start)