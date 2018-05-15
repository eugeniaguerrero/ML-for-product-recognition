from keras.applications.inception_v3 import InceptionV3
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from src.callbacks import *
from src.DATA_PREPARATION.folder_manipulation import *
from src.NN_MODELS.common_network_operations import *

class INCEPTION_V3(object):
    def __init__(self,lr=0.0001,cached_model= None,IM_HEIGHT=100,IM_WIDTH=100):
        self.model_name = "inception_v3"
        self.model_input = (1, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)
        # create the base pre-trained model
        self.base_model = InceptionV3(weights='imagenet', include_top=False)
        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(NUMBER_CLASSES, activation='softmax')(x)
        self.lr = lr
        # this is the model we will train
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        if cached_model is not None:
            self.model = load_model(cached_model)

    def train(self,train_directory_, validation_directory_,model_description,epochs,datagen,datagenval):
        self.model_name += model_description
        create_folder_structure()

        calls_ = logs()

        train_generator = datagen.flow_from_directory(
            train_directory_,
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical")

        validate_generator = datagenval.flow_from_directory(
            validation_directory_,
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical")  # CHANGE THIS!!!

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])

        # train the model on the new data for a few epochs


        self.model.fit_generator(train_generator, validation_data=validate_generator,callbacks=[calls_.json_logging_callback,
                                                         calls_.slack_callback,keras.callbacks.TerminateOnNaN(),
                                                            get_model_checkpoint(),get_Tensorboard()],epochs=epochs)
        print("Model saved to " + os.path.join(MODEL_SAVE_FOLDER, self.model_name + "_Part_1" + '.hdf5'))
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.model.save(os.path.join(MODEL_SAVE_FOLDER, str(self.model_name + "_Part_1" '.hdf5')))
        clean_up(self.model_name + "_Part_1")

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(self.base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from keras.optimizers import SGD
        self.model.compile(optimizer=SGD(self.lr, momentum=0.9), loss='categorical_crossentropy',metrics = ['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        self.model.fit_generator(train_generator, validation_data=validate_generator,callbacks=[calls_.json_logging_callback,
                                                                        calls_.slack_callback,keras.callbacks.TerminateOnNaN(),
                                                                            get_model_checkpoint(),get_Tensorboard()],epochs=epochs)
        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(MODEL_SAVE_FOLDER, self.model_name + "_Part_2" + '.hdf5'))
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + "_Part_2" '.hdf5')))

        clean_up(self.model_name + "_Part_2")

    def predict(self,input_data):
        input_data = input_data / 255
        predictions = self.model.predict(input_data, verbose=False)
        return np.array(predictions[0])

    def return_weights(self,layer):
        return self.model.layers[layer].get_weights()
