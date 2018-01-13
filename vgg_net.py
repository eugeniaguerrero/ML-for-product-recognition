import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model
from callbacks import *
from common import *

class NN(object):
    def __init__(self,cached_model= None):
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

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])


    def train(self,directory_,model_name,epochs):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        calls_ = logs()

        train_generator = train_datagen.flow_from_directory(
            directory_,
            target_size=(100, 100),
            batch_size=32,
            class_mode="categorical")  # CHANGE THIS!!!

        self.model.fit_generator(train_generator, callbacks=[calls_.json_logging_callback ],epochs=10)


        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir, "models", model_name + '.hdf5'))
        self.model.save("models/" + model_name + '.hdf5')

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

        folders = os.listdir(directory_)
        folder_present = False

        for d in folders:
            if os.path.isdir(os.path.join(directory_, d)):
                folder = d
                folder_present = True
                break
        if not folder_present:
            print("Could not find any folders/categories!")
        files = os.listdir(os.path.join(directory_, d))

        image_present = False

        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image = file
                image_present = True
                break
        if not image_present:
            print("Could not find any Images!")
        filepath = directory_ + '/' + folder + '/' + image
        import cv2
        img = cv2.imread(filepath)
        resized_image = np.expand_dims(cv2.resize(img, (IM_HEIGHT, IM_WIDTH)),axis = 0)
        print(resized_image.shape)
        predictions = self.predict(resized_image)

        if np.max(predictions) - np.min(predictions) > 0.1:
            print("Starting with a pre-trained model")
        else:
            print("Starting without a pre-trained model")
        print("Initial predictions are:")
        print(predictions)
        self.train(directory_,'debugging_model',10)



