import src.DATA_PREPARATION.data_generator as test
import numpy as np

import unittest

gen = test.DataGenerator(dir='testdata')


class ComponentTestCase(unittest.TestCase):

    #Testing whether get_files function delivers correct amount of elements in list for x-axis
    def test_get_files(self):
        x,y = gen.get_files('testdata')
        exp_n_list = 20
        self.assertEqual(exp_n_list, len(x))

    #Testing whether get_files function delivers correct amount of elements in list for x-axis
    def test_get_files2(self):
        x,y = gen.get_files('testdata')
        exp_shape_y = (20,10)
        self.assertEqual(exp_shape_y, y.shape)


    #Sequence_length = 1: Testing whether test_generator crops the right amount of pictures
    def test_generate_sqnc_1(self):
        #expected output shape for pictures: 4 pictures in each folder and two batches = 8 pictures;
        #picture dimensions are (100,100,3), therefore expected output format is (8,100,100,3)
        #expected label form is therefore in shape (8,10)
        datagen = test.DataGenerator(dir='testdata',
                                     batch_size=2,
                                     shuffle=True,
                                     sequence_length=1)
        exp_dims_picture = (8,100,100,3)
        exp_dims_labels = (8,10)
        gen = datagen.generate()
        test_in = gen.__next__()
        self.assertEqual(exp_dims_picture, test_in[0].shape)
        self.assertEqual(exp_dims_labels, test_in[1].shape)

    def test_generate_sqnc_4(self):
        #expected output shape for pictures: 4 pictures in each folder and two batches = 8 pictures;
        #picture dimensions are (100,100,3), therefore expected output format is (8,100,100,3)
        #expected label form is therefore in shape (8,10)
        datagen = test.DataGenerator(dir='testdata',
                                     batch_size=2,
                                     shuffle=True,
                                     sequence_length=4)
        exp_dims_picture = (2,4,100,100,3)
        exp_dims_labels = (2,10)
        gen = datagen.generate()
        test_in = gen.__next__()
        self.assertEqual(exp_dims_picture, test_in[0].shape)
        self.assertEqual(exp_dims_labels, test_in[1].shape)

if __name__ == '__main__':
    unittest.main()