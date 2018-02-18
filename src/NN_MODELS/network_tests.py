import numpy as np
from src.common import *
from src.DATA_PREPARATION.folder_manipulation import *

def debug(NN):
    # Test 1 check if untrained model returns uniform predictions

    images = np.random.random_sample(NN.model_input) * 255
    predictions = NN.predict(images)
    print("Initial predictions are:" + str(predictions))
    if not PRETRAINED_MODEL:
        assert np.max(predictions) - np.min(predictions) < 0.15,"Untrained predictions not evenly distributed."
        print("Starting with a pre-trained model")
    else:
        print("Starting without a pre-trained model")

    # Test 2 see if accuracy goes very quickly to 1 on 1 image
    NN.train(DEBUG_FOLDER, DEBUG_FOLDER, 'debug_model', 10)
    image_name = os.path.join("0","1.jpg")
    print(os.path.join(DEBUG_FOLDER,image_name))
    image = get_image(os.path.join(DEBUG_FOLDER,image_name))
    predictions = []
    for i in range(100):
        predictions.append(np.argmax(NN.predict(image)))
    assert max(predictions) == min(predictions) & predictions[0] == 0, "Network did not learn to classify one image"
    print("TESTING COMPLETE - commencing training...")


def find_incorrect_classifications(directory_,NN):
    incpred = "incorrect_predictions"
    if not os.path.exists(incpred):
        os.makedirs(incpred)

    # Test 1 check if untrained model returns uniform predictions
    folders = get_folders(directory_)
    category = 0
    import shutil
    for folder in folders:

        if not os.path.exists(os.path.join(incpred, folder)):
            os.makedirs(os.path.join(incpred, folder))

        image_list = get_image_names(os.path.join(directory_, folder))
        for image in image_list:
            filepath = os.path.join(directory_, folder, image)
            resized_image = get_image(filepath)
            predictions = NN.predict(resized_image)

            if np.argmax(predictions) != category:
                fileto = os.path.join(incpred, folder, image)
                shutil.copyfile(filepath, fileto)
        category = category + 1