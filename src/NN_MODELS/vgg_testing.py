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
    def __init__(self,lr=2,conv1_size = 6,conv2_size = 7,dense_size = 8,decay = 6,moment = 0.9,cached_model= None,IM_HEIGHT=100,IM_WIDTH=100):
        self.model_name = "vgg_net"
        self.model_input = (1, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)
        self.im_height = IM_HEIGHT
        self.im_width = IM_WIDTH
        self.model = Sequential()
        #SORT OUT THE DIMENSIONS
        conv1_size = int(2**conv1_size)
        conv2_size = int(2 ** conv2_size)
        dense_size = int(2** dense_size)

        decay = 10 **(-decay)
        lr = 10**(-lr)
        print(lr)
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Conv2D(conv1_size, (3, 3), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        self.model.add(Conv2D(conv1_size, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(conv2_size, (3, 3), activation='relu'))
        self.model.add(Conv2D(conv2_size, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(dense_size, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(NUMBER_CLASSES, activation='softmax'))

        if cached_model is not None:
            self.model = load_model(cached_model)

        sgd = SGD(lr, decay=decay, momentum=moment, nesterov=True)
        #self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])

    def train(self,train_directory_, validation_directory_,model_description,epochs,datagen,datagenval):
        self.model_name += model_description
        create_folder_structure()

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        calls_ = logs()

        print("Image dimensions are : " + str(self.im_height))

        train_generator = datagen.flow_from_directory(
            train_directory_,
            target_size=(self.im_height, self.im_width),
            batch_size=BATCH_SIZE,
            class_mode="categorical")

        validate_generator = datagenval.flow_from_directory(
            validation_directory_,
            target_size=(self.im_height, self.im_width),
            batch_size=BATCH_SIZE,
            class_mode="categorical")  # CHANGE THIS!!!

        self.model.fit_generator(train_generator, validation_data=validate_generator,callbacks=[calls_.json_logging_callback,
                                                             calls_.slack_callback,
                                                             keras.callbacks.TerminateOnNaN(),
                                                             get_model_checkpoint(),get_Tensorboard()],epochs=epochs)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir, MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
        self.model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
        clean_up(self.model_name)



    def predict(self,input_data):
        input_data = input_data / 255
        predictions = self.model.predict(input_data, verbose=False)
        return np.array(predictions[0])


    def return_weights(self,layer):
        return self.model.get_weights(layer)
