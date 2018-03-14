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
    def __init__(self,output = True,lr=2,conv1_size = 5,conv2_size = 6,dense_size = 8,decay = 6,moment = 0.9,cached_model= None,IM_HEIGHT=100,IM_WIDTH=100):
        self.model_name = "vgg_net"
        self.output = output
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
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])

    def train(self,train_directory_, validation_directory_,model_description,epochs,datagen,datagenval):
        self.model_name += model_description

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
                                                             keras.callbacks.ModelCheckpoint(filepath=INTERMEDIATE_FILE,
                                                                                             monitor='val_loss',
                                                                                             verbose=0,
                                                                                             save_best_only=False,
                                                                                             save_weights_only=False,
                                                                                             mode='auto', period=1),
                                                             keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_FOLDER,
                                                                                         histogram_freq=0,
                                                                                         batch_size=BATCH_SIZE,
                                                                                         write_graph=True,
                                                                                         write_grads=False,
                                                                                         write_images=True,
                                                                                         embeddings_freq=0,
                                                                                         embeddings_layer_names=None,
                                                                                         embeddings_metadata=None)],epochs=epochs)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir, MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
        clean_up(self.model_name)



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


    def return_weights(self,layer):
        return self.model.get_weights(layer)


