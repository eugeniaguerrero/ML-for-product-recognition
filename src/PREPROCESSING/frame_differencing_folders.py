import inspect, sys, os, shutil, time, cv2, numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
from skimage.measure import compare_ssim
from skimage.util.shape import view_as_blocks
from src.DATA_PREPARATION.folder_manipulation import *
from src.PREPROCESSING.histogram_equalisation import *


# def mse(imageA, imageB):
#     # sum of the squared difference between the two images;
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#     # return the MSE, the lower the error, the more "similar"
#     return err

def square_border(img):
    if img.shape[0] > img.shape[1]:
        border_size = int((img.shape[0] - img.shape[1])/2)
        bordered_img = cv2.copyMakeBorder(img, top=0, bottom=0, left=border_size, right=border_size,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        border_size = int((img.shape[1] - img.shape[0])/2)
        bordered_img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=0, right=0,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return bordered_img

def img_exception(img):
    height = img.shape[0]
    width = img.shape[1]

    # if checks whether the image has not been cropped (i.e. not frame differenced)
    if (height == RAW_HEIGHT) and (width == RAW_WIDTH):
        return True

    # any image cropped to below this size will be raised as an exception
    if (height < 200) or (width < 200):
        return True

    return False


def kernel_compare(individual_product, next_product):
    region_size = 64

    im_h = individual_product.shape[0]
    im_w = individual_product.shape[1]

    bl_h, bl_w = region_size, region_size

    # 1 means the pics are identical. Try 085
    ssim_threshold = 0.85

    white = 255

    gray_img = np.zeros((im_h, im_w))
    gray_img.fill(white)
    gray_img = gray_img.astype(np.uint8)

    # color_img = np.zeros((im_h, im_w, 3))
    # color_img.fill(white)
    # color_img = color_img.astype(np.uint8)

    grayA = cv2.cvtColor(individual_product, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(next_product, cv2.COLOR_BGR2GRAY)

    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):

            grayA_kernel = grayA[row:row + bl_h, col:col + bl_w]
            grayB_kernel = grayB[row:row + bl_h, col:col + bl_w]

            (ssim_score, diff) = compare_ssim(grayA_kernel, grayB_kernel, full=True)
            # checking to see if the pictures are different
            if (ssim_score < ssim_threshold):
                gray_img[row:row + bl_h, col:col + bl_w] = grayA[row:row + bl_h, col:col + bl_w]
                #color_img[row:row + bl_h, col:col + bl_w] = individual_product[row:row + bl_h, col:col + bl_w]


    thresh = 254.9
    threshold_img = cv2.threshold(gray_img, thresh, white, cv2.THRESH_BINARY_INV)[1]

    # grayA_blocks = view_as_blocks(grayA, (region_size, region_size))
    # grayB_blocks = view_as_blocks(grayB, (region_size, region_size))
    #
    # grayA_blocks_reshaped = np.reshape(grayA_blocks, (-1, region_size, region_size))
    # grayB_blocks_reshaped = np.reshape(grayB_blocks, (-1, region_size, region_size))
    #
    # ssim_array = map(lambda v1, v2: compare_ssim(v1, v2), grayA_blocks_reshaped, grayB_blocks_reshaped)
    #
    # #ssim_array = np.fromiter(ssim_array, dtype=np.uint8)
    # ssim_array = np.array(list(ssim_array))
    #
    # ssim_array_reshape = np.reshape(ssim_array, (int(im_h / region_size), int(im_w / region_size)))
    #
    # ssim_array_reshape = np.where(ssim_array_reshape < ssim_threshold, 1, 0)
    # ssim_array_full_size = np.kron(ssim_array_reshape, np.ones((region_size, region_size)))
    #
    # gray_img = np.multiply(ssim_array_full_size, grayA)
    #
    # thresh = 0
    # color = 255
    # threshold_img = cv2.threshold(gray_img, thresh, color, cv2.THRESH_BINARY)[1]
    #
    # threshold_img = threshold_img.astype(np.uint8)

    _, contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if we have no contours then there is no difference between frames
    if contours == []:
        return individual_product

    largest_area = sorted(contours, key=cv2.contourArea)[-1]
    # print("Largest:", cv2.contourArea(largest_area))
    # next_largest_area = sorted(contours, key=cv2.contourArea)[-2]
    # print("2nd Largest:", cv2.contourArea(next_largest_area))

    x, y, w, h = cv2.boundingRect(largest_area)

    cropped_individual_product = individual_product[y: y + h, x: x + w]

    histogram_block_img = CLAHE_equalisation(cropped_individual_product)

    # blank_square = np.zeros((h, w))
    # blank_square.fill(0)
    #
    # histogram_grey_block_img = cv2.cvtColor(histogram_block_img, cv2.COLOR_BGR2GRAY)
    #
    # kernel = np.ones((5, 5), np.uint8)
    # # morphology removes noise from the picture
    # close_operated_image = cv2.morphologyEx(histogram_grey_block_img, cv2.MORPH_CLOSE, kernel)
    #
    # # OTSU method finds optimal threshold
    # _, thresholded = cv2.threshold(close_operated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # reduces noise by taking the median value in the kernel
    # median = cv2.medianBlur(thresholded, 5)
    #
    # _, contours, _ = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # blank_square = cv2.drawContours(blank_square, contours, -1, (100, 0, 255), 2)
    #
    # cv2.imshow("mypic", cropped_block_img)
    # cv2.waitKey(0)

    # move to after we look for exceptions

    return histogram_block_img


def frame_difference_collection(source, destination, exceptions):
    i = 0
    num_of_pics = 0

    # find the number of pictures in the video set
    for filename in sorted(os.listdir(source)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            num_of_pics = num_of_pics + 1

    # store the pictures in an array
    pictures = np.empty((num_of_pics, RAW_HEIGHT, RAW_WIDTH, 3), dtype=np.uint8)
    for filename in sorted(os.listdir(source)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(source, filename)
            image = cv2.imread(filepath)
            # pictures[i, ...] = image
            if image.shape[0] == RAW_HEIGHT and image.shape[1] == RAW_WIDTH:
                pictures[i] = image
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

        # here we check whether frame differencing has detected movement
        if img_exception(new_img):
            file_name = 'exception' + str(index) + '_' + filename
            cv2.imwrite(os.path.join(exceptions, file_name), individual_product)
        else:
            new_img = square_border(new_img)
            # note that we still store the remaining video collection
            file_name = str(index) + '_' + filename
            original_file_name = str(index) + '_original_' + filename

            cv2.imwrite(os.path.join(destination, file_name), new_img)
            # include for comparison
            # cv2.imwrite(os.path.join(destination, original_file_name), individual_product)
        index += 1


def main_diff(folder_set, data_path):
    print(data_path)
    my_count = 0

    # ensure that folder exist
    for folder in folder_set:
        my_folder = os.path.join(data_path, folder)
        assert (os.path.exists(my_folder))

    # loop on training, validation and testing
    for folder in folder_set:
        difference_set = 'FD_' + folder
        my_folder = os.path.join(data_path, difference_set)

        if os.path.exists(my_folder):
            shutil.rmtree(my_folder)
        os.makedirs(my_folder)

        data_set = os.path.join(data_path, folder)
        class_folders = get_folders(data_set)
        assert (len(class_folders) > 0)

        exceptions = 'Exceptions_' + folder
        my_exceptions = os.path.join(data_path, exceptions)

        if os.path.exists(my_exceptions):
            shutil.rmtree(my_exceptions)
        os.makedirs(my_exceptions)

        # loop on class folders
        for class_set in class_folders:
            class_folder = os.path.join(data_set, class_set)

            diff_class_folder = os.path.join(data_path, difference_set, class_set)
            os.makedirs(diff_class_folder)

            exceptions_class_folder = os.path.join(data_path, exceptions, class_set)
            os.makedirs(exceptions_class_folder)

            product_video_set = get_folders(class_folder)
            assert (len(product_video_set) > 0)

            # loop on video sets
            for video in product_video_set:
                video_folder = os.path.join(class_folder, video)
                diff_video_folder = os.path.join(diff_class_folder, video)
                os.makedirs(diff_video_folder)
                frame_difference_collection(video_folder, diff_video_folder, exceptions_class_folder)
                my_count = my_count + 1
                    # avg_time = (time.time() - start_run)/my_count
                    # print("Avg time per set:", avg_time)
    return True

if __name__ == "__main__":
    start_run = time.time()
    print("Start Run")
    # The directory where the augmented images are going to be saved

    dir = os.getcwd()
    src_dir = os.path.dirname(dir)
    Group_dir = os.path.dirname(src_dir)
    data_folder = 'DATA'
    data_path = os.path.join(Group_dir, data_folder)

    #my_folders = ['test_data', 'training_data', 'validation_data']
    my_folders = ['testing']

    main_diff(my_folders, data_path)

    end_run = time.time()
    print("Total Run Time (mins): ", (end_run - start_run)/60)



