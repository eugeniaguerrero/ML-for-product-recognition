import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
import src.DATA_PREPARATION.data_generator as test
import numpy as np
from src.common import *

import unittest

dat = os.path.join(currentdir,'testdata')

gen = test.DataGenerator(dir= dat)

class ComponentTestCase(unittest.TestCase):

    #Testing whether get_files function delivers correct amount of elements in list for x-axis
    def test_get_files(self):
        #Expected: 10 class folders with 2 inside = 20 folders in total
        exp_n_list = 20
        x,y = gen.get_files(dat)
        self.assertEqual(exp_n_list, len(x))

    #Testing whether get_files function delivers correct amount of elements in list for y-axis
    def test_get_files2(self):
        #Expected: 1 label per folder with 10 classes = (20,10)
        x,y = gen.get_files(dat)
        exp_shape_y = (20,10)
        self.assertEqual(exp_shape_y, y.shape)

    #Sequence_length = 1: Testing whether test_generator crops the right amount of pictures
    def test_generate_sqnc_1(self):
        #Expected shape x: (8,100,100,3) due to batch_size = 2, sequence_length = 1 and 4 pictures in one folder
        #Expected shape y: (8,10) due to 8 pictures and sequence_length = 1
        datagen = test.DataGenerator(dir=dat,
                                     batch_size=2,
                                     shuffle=True,
                                     time_distributed=False,
                                     sequence_length=FILES_IN_FOLDER)
        exp_dims_picture = (2,100,100,3)
        exp_dims_labels = (2,10)
        gen = datagen.generate()
        test_in = gen.__next__()
        self.assertEqual(exp_dims_picture, test_in[0].shape)
        self.assertEqual(exp_dims_labels, test_in[1].shape)

    def test_generate_sqnc_4(self):
        #Expected shape x: (8,100,100,3) due to batch_size = 2, sequence_length = 1 and 4 pictures in one folder
        #Expected shape y: (8,10) due to 8 pictures and sequence_length = 1
        datagen = test.DataGenerator(dir=dat,
                                     batch_size=2,
                                     shuffle=True,
                                     time_distributed=True,
                                     sequence_length=FILES_IN_FOLDER)

        exp_dims_picture = (2,4,100,100,3)
        exp_dims_labels = (2,10)
        gen = datagen.generate()
        test_in = gen.__next__()
        self.assertEqual(exp_dims_picture, test_in[0].shape)
        self.assertEqual(exp_dims_labels, test_in[1].shape)

    def test_generate_sqnc_4(self):
        #10 classes * 2 folders * 4 pictures = 80 pictures
        #Getting 4 pictures per batch with 2 batches = 8 pictures per step
        #80 pictures / 8 pictures = 10 steps
        datagen = test.DataGenerator(dir=dat,
                                     batch_size=2,
                                     shuffle=True,
                                     time_distributed=True,
                                     sequence_length=FILES_IN_FOLDER)

        self.assertEqual(10, datagen.batches_per_epoch)


    #Testing whether datagenerator identifies correct number of steps per epoch
    def test_generate_sqnc_1(self):
        #10 classes * 2 folders * 4 pictures = 80 pictures
        #Getting 1 pictures per batch with 2 batches = 2 pictures per step
        #80 pictures / 2 pictures = 40 steps
        datagen = test.DataGenerator(dir=dat,
                                     batch_size=2,
                                     shuffle=True,
                                     time_distributed=False,
                                     sequence_length=FILES_IN_FOLDER)

        self.assertEqual(40, datagen.batches_per_epoch)

    def test_generate_sqnc_4_loop(self):
        #Expected shape x: (8,100,100,3) due to batch_size = 2, sequence_length = 1 and 4 pictures in one folder
        #Expected shape y: (8,10) due to 8 pictures and sequence_length = 1
        datagen = test.DataGenerator(dir=dat,
                                     batch_size=2,
                                     shuffle=True,
                                     time_distributed=True,
                                     sequence_length=FILES_IN_FOLDER)


        exp_dims_picture = (2,4,100,100,3)
        exp_dims_labels = (2,10)

        gen = datagen.generate()

        epoch_counter = 1
        batch_counter = 0

        for i in range (datagen.batches_per_epoch*3):
            test_in = gen.__next__()
            self.assertEqual(epoch_counter, datagen.epoch_number)
            self.assertEqual(exp_dims_picture, test_in[0].shape)
            self.assertEqual(exp_dims_labels, test_in[1].shape)
            self.assertEqual(exp_dims_labels, test_in[1].shape)
            batch_counter += 1

            if batch_counter == datagen.batches_per_epoch:
                epoch_counter += 1
                batch_counter = 0



    def test_generate_sqnc_1_loop(self):
        #Expected shape x: (8,100,100,3) due to batch_size = 2, sequence_length = 1 and 4 pictures in one folder
        #Expected shape y: (8,10) due to 8 pictures and sequence_length = 1
        datagen = test.DataGenerator(dir=dat,
                                     batch_size=2,
                                     shuffle=True,
                                     time_distributed=False,
                                     sequence_length=FILES_IN_FOLDER)

        exp_dims_picture = (2,100,100,3)
        exp_dims_labels = (2,10)


        gen = datagen.generate()

        epoch_counter = 1
        batch_counter = 0

        for i in range (datagen.batches_per_epoch*3):
            test_in = gen.__next__()
            self.assertEqual(epoch_counter, datagen.epoch_number)
            self.assertEqual(exp_dims_picture, test_in[0].shape)
            self.assertEqual(exp_dims_labels, test_in[1].shape)
            self.assertEqual(exp_dims_labels, test_in[1].shape)
            batch_counter += 1

            if batch_counter == datagen.batches_per_epoch:
                epoch_counter += 1
                batch_counter = 0

if __name__ == '__main__':
    unittest.main()