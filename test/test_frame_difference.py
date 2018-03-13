import unittest
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from src.PREPROCESSING.frame_differencing_folders import *
from src.DATA_PREPARATION.folder_manipulation import *
import cv2

dir = os.getcwd()
Group_dir = os.path.dirname(dir)
test_folder = 'group-project-back-end/test'
test_subfolder = 'fd_test_data'
data_path = os.path.join(Group_dir, test_folder, test_subfolder)

class arun_test(unittest.TestCase):

    # initialization logic
    # code that is executed before each test
    def setUp(self):
        pass

    # code that is executed after each test
    def tearDown(self):
        pass

        # check if e.g. test data folder does not exist
    def test_folder_does_not_exist(self):
        list = ['this_is_not_a_folder']
        with self.assertRaises(AssertionError):
            main_diff(list, data_path)

    # check if method functions correctly when data is correct
    def test_correct_folder_setup_exists(self):
        list = ['sample1']
        print("MY PATH", data_path)
        self.assertTrue(main_diff(list, data_path))

    # raises an error if a folder has no data
    def test_empty_folder(self):
        list = ['empty_folder']
        with self.assertRaises(AssertionError):
            main_diff(list, data_path)

    # raises an error if a class has no data
    def test_empty_subfolder(self):
        list = ['empty_subfolder']
        with self.assertRaises(AssertionError):
            main_diff(list, data_path)

    # checks whether similar images are detected as duplicated
    def test_duplicate_pictures_subfolder(self):
        list = ['image_duplicates']
        main_diff(list, data_path)
        exception_folder = 'Exceptions_' + list[0]
        exceptions_path = os.path.join(data_path, exception_folder, 'my_class')
        self.assertTrue(len(get_image_names(exceptions_path)) == 10)

    # checks whether different images are not marked as exceptions
#    def test_different_pictures_subfolder(self):
#        list = ['non_duplicates']
#        main_diff(list, data_path)
#        exception_folder = 'Exceptions_' + list[0]
#        exceptions_path = os.path.join(data_path, exception_folder, 'my_class')
#        self.assertTrue(len(get_image_names(exceptions_path)) == 0)

    # check whether a non square image is converted to a square image
    def test_square_image(self):
        my_image_path = os.path.join(data_path, 'non_square_pic.jpg')
        my_image = cv2.imread(my_image_path)
        my_square_image = square_border(my_image)
        self.assertTrue(my_square_image.shape[0] == my_square_image.shape[1])

if __name__ == "__main__":
    unittest.main()
