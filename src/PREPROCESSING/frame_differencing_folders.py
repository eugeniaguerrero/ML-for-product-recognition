import inspect, sys, os, shutil, time, cv2, numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
from skimage.measure import compare_ssim
from src.DATA_PREPARATION.folder_manipulation import *
from src.PREPROCESSING.histogram_equalisation import *


# returns a square image with black pixels extending the border of the shortest dimension
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

# checks whether the image has been reduced in size and whether it has been reduced below a threshold
# if true, returns an exception
def img_exception(img):
    height = img.shape[0]
    width = img.shape[1]

    if (height == RAW_HEIGHT) and (width == RAW_WIDTH):
        return True

    if (height < 200) or (width < 200):
        return True

    return False

# returns a cropped image around the dominant moving area of the image
def kernel_compare(individual_product, next_product):

    # pizel size the side of the square filter we pass over the image
    region_size = 64

    im_h = individual_product.shape[0]
    im_w = individual_product.shape[1]

    bl_h, bl_w = region_size, region_size

    # set the structural simularity threshold of the function
    # 1 means the pics are identical. 0 means no simularity
    # Try 0.85
    ssim_threshold = 0.85

    # we apply SSIM to the image converted to grey scale
    # if there is sufficient difference between the consecutive images for a section of the image, we add this square
    white = 255
    gray_img = np.zeros((im_h, im_w))
    gray_img.fill(white)
    gray_img = gray_img.astype(np.uint8)

    grayA = cv2.cvtColor(individual_product, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(next_product, cv2.COLOR_BGR2GRAY)

    # we pass over a square kernel across two consecutive images in the video sequence
    # if there is a structural difference between the two images, we keep these pixels
    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):

            grayA_kernel = grayA[row:row + bl_h, col:col + bl_w]
            grayB_kernel = grayB[row:row + bl_h, col:col + bl_w]

            (ssim_score, diff) = compare_ssim(grayA_kernel, grayB_kernel, full=True)
            # checking to see if the pictures are different
            if (ssim_score < ssim_threshold):
                gray_img[row:row + bl_h, col:col + bl_w] = grayA[row:row + bl_h, col:col + bl_w]

    # we use contouring to find the largest connected area of moving pixels
    thresh = 254.9
    threshold_img = cv2.threshold(gray_img, thresh, white, cv2.THRESH_BINARY_INV)[1]
    _, contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if we have no contours then there is no difference between frames (we have kept no pixels)

    if contours == []:
        return individual_product

    largest_area = sorted(contours, key=cv2.contourArea)[-1]

    # we return the edge points of the largest rectangle and then crop on these points
    x, y, w, h = cv2.boundingRect(largest_area)

    cropped_individual_product = individual_product[y: y + h, x: x + w]

    return cropped_individual_product


# this function frame differences a video of set of n images
# we also apply histogram equalisation
# if frame differencing has not detected any movement then we add it as an exception
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

        kernelized_img = kernel_compare(individual_product, next_product)

        histogram_img = CLAHE_equalisation(kernelized_img)

        new_img = histogram_img[0, :, :, :]

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
                avg_time = (time.time() - start_run)/my_count
                print("Avg time per set:", avg_time)
    return True

if __name__ == "__main__":
    start_run = time.time()
    print("Start Run")

    dir = os.getcwd()
    src_dir = os.path.dirname(dir)
    Group_dir = os.path.dirname(src_dir)
    data_folder = 'DATA'
    data_path = os.path.join(Group_dir, data_folder, 'raw_ambient')
    my_folders = ['validation_data', 'test_data','training_data']
    print(data_path)
    main_diff(my_folders, data_path)

    end_run = time.time()
    print("Total Run Time (mins): ", (end_run - start_run)/60)