from src.NN_MODELS.vgg_net import *
from src.NN_MODELS.inception_v3 import *
from src.DATA_PREPARATION.folder_manipulation import *
import unittest

class TestNetwork(unittest.TestCase):
    def test_training(self):
        # Test 1 check if untrained model returns uniform predictions
        Networks = [VGG(output = True),INCEPTION_V3(output = True)]

        for NN in Networks:

            # store weights before
            before_softmax = NN.return_weights(-1)

            # Test 2 see if accuracy goes very quickly to 1 on 1 image
            NN.train(DEBUG_FOLDER, DEBUG_FOLDER, 'debug_model', 10)

            #CHECK model weights have changed
            after_softmax = NN.return_weights(-1)

            # check that something has changed
            print(str(before_softmax) + " , " + str(after_softmax))
            self.assertFalse(np.array_equal(before_softmax, after_softmax))

            image_name = os.path.join("0","1.jpg")

            print(os.path.join(DEBUG_FOLDER,image_name))
            image = get_image(os.path.join(DEBUG_FOLDER,image_name))
            predictions = []
            for i in range(100):
                predictions.append(np.argmax(NN.predict(image)))
            self.assertTrue(max(predictions) == min(predictions) & predictions[0] == 0)
            print("TESTING COMPLETE - commencing training...")

    def test_untrainednet(self):

        Networks = [VGG(output = False), INCEPTION_V3(output = False)]

        for NN in Networks:
            images = np.random.random_sample(NN.model_input) * 255
            predictions = NN.predict(images)
            print("Initial predictions are:" + str(predictions))
            if not PRETRAINED_MODEL:
                self.assertTrue(np.max(predictions) - np.min(predictions) < 0.15)
                print("Starting with a pre-trained model")
            else:
                print("Starting without a pre-trained model")


if __name__ == '__main__':
    unittest.main()