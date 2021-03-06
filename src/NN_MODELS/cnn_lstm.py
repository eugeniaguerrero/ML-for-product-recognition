from keras.layers import Input, LSTM
from keras.models import Model
from src.DATA_PREPARATION.data_generator import *
import keras
from keras import backend as K
 #set learning phase
K.set_learning_phase(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from src.callbacks import *
from src.DATA_PREPARATION.folder_manipulation import *
from src.NN_MODELS.common_network_operations import *
from keras.layers import TimeDistributed

class CNN_LSTM(object):
    def __init__(self,output = True,lr=0.01,cached_model= None):
        self.model_name = "vgg_net"
        self.output = output
        self.model_input = (1,IMAGES_PER_FOLDER,IM_HEIGHT,IM_WIDTH,NUMBER_CHANNELS)

        self.cnnmodel = Sequential()
        self.cnnmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)))
        self.cnnmodel.add(Conv2D(32, (3, 3), activation='relu'))
        self.cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnnmodel.add(Dropout(0.25))
        self.cnnmodel.add(Conv2D(64, (3, 3), activation='relu'))
        self.cnnmodel.add(Conv2D(64, (3, 3), activation='relu'))
        self.cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnnmodel.add(Dropout(0.25))
        self.cnnmodel.add(Flatten())

        video_input = Input(shape=(IMAGES_PER_FOLDER, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS))

        encoded_frame_sequence = TimeDistributed(self.cnnmodel)(video_input) # the output will be a sequence of vectors
        encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be one vector

        output = Dense(NUMBER_CLASSES, activation='softmax')(encoded_video)
        self.model = Model(inputs=[video_input], outputs=output)

        if cached_model is not None:
            self.model = load_model(cached_model)

        sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train(self,train_directory_, validation_directory_,model_description,epochs):
        self.model_name += model_description
        create_folder_structure()

        params_val = {'dir': validation_directory_,
                  'batch_size': 36,
                  'shuffle': True,
                  'sequence_length' : 12,'time_distributed' : True}

        validation_generator = DataGenerator(**params_val)
        validate_gen = validation_generator.generate()

        params_train = {'dir': train_directory_,
                      'batch_size': 36,
                      'shuffle': True,
                        'sequence_length': 12,'time_distributed' : True}

        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()

        #CHECKS!################
        test_in = train_gen.__next__()
        test_in_val = validate_gen.__next__()

        steps_per_epoch_ =  train_generator.batches_per_epoch
        validation_steps_ = validation_generator.batches_per_epoch
        ##########################

        calls_ = logs()
        self.model.fit_generator(train_gen, validation_data=validate_gen,
                                 callbacks=[calls_.json_logging_callback,
                                            calls_.slack_callback,
                                            get_model_checkpoint(),get_Tensorboard()], steps_per_epoch =steps_per_epoch_,
                                                                                    validation_steps=steps_per_epoch_, epochs=epochs)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir,MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
        clean_up(self.model_name)

    def predict(self,input_data):
        K.set_learning_phase(0)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # CHANGED THIS!!!!
        input_data = input_data / 255
        predictions = self.model.predict(input_data, verbose=False)
        return np.array(predictions[0])


    def return_weights(self,layer):
        return self.model.layers[layer].get_weights()
