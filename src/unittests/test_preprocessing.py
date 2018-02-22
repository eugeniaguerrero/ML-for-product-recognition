import src.PREPROCESSING.preprocessing as test
import numpy as np

import unittest

matrix_input = np.random.normal(size = (4, 100, 100, 3))
benchmark_output_shape = np.zeros((4, 100, 100, 3)).shape

class ComponentTestCase(unittest.TestCase):

    #Tests the overall preprocessing function
    #Includes testing whether the shape dimensions are correct for random_rotation, random_zoom, flip_horizontal and flip_vertical
    def test_preprocessing_shape(self):
            preprocessing = test.Preprocessing(rotation = True,
                                               rotation_degrees = 30,
                                               zoom = True,
                                               zoom_max = 0.9,
                                               horizontal_flip = True,
                                               vertical_flip = True,
                                               histogram_equalisation = False)

            output_testfunction_shape = preprocessing.preprocess_images(matrix_input).shape
            self.assertEqual(output_testfunction_shape, benchmark_output_shape)

    #test whether total sum of vector does not change if matrix is flipped
    def test_preprocessing_matrix_content(self):
        preprocessing = test.Preprocessing(horizontal_flip = True,
                                           vertical_flip = True)

        output_testfunction = preprocessing.preprocess_images(matrix_input)
        x = abs(np.sum(output_testfunction) - np.sum(matrix_input))
        self.assertTrue(x < 1e-10)



if __name__ == '__main__':
    unittest.main()