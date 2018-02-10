from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from src.callbacks import *
from src.DATA_PREPARATION.folder_manipulation import *
from src.NN_MODELS.common_network_operations import *

class VGG(object):
    def __init__(self,lr=0.01,cached_model= None):
        self.model_name = "vgg_net"
        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(NUMBER_CLASSES, activation='softmax'))

        if cached_model is not None:
            self.model = load_model(cached_model)

        sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])

    def train(self,train_directory_, validation_directory_,model_description,epochs):
        self.model_name += model_description
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        calls_ = logs()

        train_generator = datagen.flow_from_directory(
            train_directory_,
            target_size=(100, 100),
            batch_size=32,
            class_mode="categorical")

        validate_generator = datagen.flow_from_directory(
            validation_directory_,
            target_size=(100, 100),
            batch_size=32,
            class_mode="categorical")  # CHANGE THIS!!!

        self.model.fit_generator(train_generator, validation_data=validate_generator,callbacks=[calls_.json_logging_callback,
                                                             calls_.slack_callback,
                                                             keras.callbacks.TerminateOnNaN(),
                                                             keras.callbacks.ModelCheckpoint(filepath=INTERMEDIATE_FILE,
                                                                                             monitor='val_loss',
                                                                                             verbose=0,
                                                                                             save_best_only=False,
                                                                                             save_weights_only=False,
                                                                                             mode='auto', period=1),
                                                             keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_FOLDER,
                                                                                         histogram_freq=0,
                                                                                         batch_size=64,
                                                                                         write_graph=True,
                                                                                         write_grads=False,
                                                                                         write_images=True,
                                                                                         embeddings_freq=0,
                                                                                         embeddings_layer_names=None,
                                                                                         embeddings_metadata=None)],epochs=10)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir, MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
        clean_up_logs(self.model_name)
        clean_up_json_logs(self.model_name)
        clean_up_models(self.model_name)



    def predict(self,input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        # CHANGED THIS!!!!
        input_data = input_data / 255
        predictions = self.model.predict(input_data, verbose=False)
        return np.array(predictions[0])


    def debug(self,directory_):

        #Test 1 check if untrained model returns uniform predictions
        folders = get_folders(directory_)
        image_list = get_image_names(os.path.join(directory_, folders[0]))
        filepath = os.path.join(directory_,folders[0],image_list[0])
        resized_image = get_image(filepath)
        predictions = self.predict(resized_image)

        if np.max(predictions) - np.min(predictions) > 0.1:
            print("Starting with a pre-trained model")
        else:
            print("Starting without a pre-trained model")
        print("Initial predictions are:")
        print(predictions)

        #Test 2 see if accuracy goes very quickly to 1 on 1 image
        self.train(directory_,'debugging_model',10)

    def find_incorrect_classifications(self,directory_):
        incpred = "incorrect_predictions"
        if not os.path.exists(incpred):
            os.makedirs(incpred)

        # Test 1 check if untrained model returns uniform predictions
        folders = get_folders(directory_)
        category = 0
        import shutil
        for folder in folders:

            if not os.path.exists(os.path.join(incpred,folder)):
                os.makedirs(os.path.join(incpred,folder))

            image_list = get_image_names(os.path.join(directory_, folder))
            for image in image_list:
                filepath = os.path.join(directory_,folder,image)
                resized_image = get_image(filepath)
                predictions = self.predict(resized_image)

                if np.argmax(predictions) != category:
                    fileto = os.path.join(incpred,folder,image)
                    shutil.copyfile(filepath,fileto)


