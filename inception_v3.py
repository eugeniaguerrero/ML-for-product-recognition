from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model
from callbacks import *
from common import *
from folder_manipulation import *

class NN(object):
    def __init__(self,cached_model= None):
        self.name = "inception_v3"
        # create the base pre-trained model
        self.base_model = InceptionV3(weights='imagenet', include_top=False)
        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(NUMBER_CLASSES, activation='softmax')(x)

        # this is the model we will train
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        if cached_model is not None:
            self.model = load_model(cached_model)

    def clean_up_logs(self):
        if not os.path.exists('old_logs'):
            os.makedirs('old_logs')
        old_logs_list = os.listdir('old_logs')
        numbers = []
        for i in old_logs_list:
            numbers.append(int(i.split('_')[0]))
        numbers = sorted(numbers)
        if len(numbers) == 0:
            count = 0
        else:
            count = numbers[-1]+1
        foldername = str(count) + '_' + self.name
        os.rename('logs', os.path.join('old_logs',foldername))
        print("Tensorboard data is in : ./old_logs/" + foldername)



    def train(self,train_directory_, validation_directory_,model_name,epochs):

        #INITIALISE DATA INPUT
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        calls_ = logs()

        train_generator = datagen.flow_from_directory(
            train_directory_,
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=32,
            class_mode="categorical")

        validate_generator = datagen.flow_from_directory(
            validation_directory_,
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=32,
            class_mode="categorical")  # CHANGE THIS!!!

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])

        # train the model on the new data for a few epochs
        self.model.fit_generator(train_generator, validation_data=validate_generator,callbacks=[calls_.json_logging_callback,
                                                             calls_.slack_callback,
                                                             keras.callbacks.TerminateOnNaN(),
                                                             keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints','intermediate.hdf5'),
                                                                                             monitor='val_loss',
                                                                                             verbose=0,
                                                                                             save_best_only=False,
                                                                                             save_weights_only=False,
                                                                                             mode='auto', period=1),
                                                             keras.callbacks.TensorBoard(log_dir='./logs',
                                                                                         histogram_freq=0,
                                                                                         batch_size=64,
                                                                                         write_graph=True,
                                                                                         write_grads=False,
                                                                                         write_images=True,
                                                                                         embeddings_freq=0,
                                                                                         embeddings_layer_names=None,
                                                                                         embeddings_metadata=None)],epochs=epochs)
        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.
        self.clean_up_logs()
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
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics = ['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        self.model.fit_generator(train_generator, validation_data=validate_generator,callbacks=[calls_.json_logging_callback,
                                                             calls_.slack_callback,
                                                             keras.callbacks.TerminateOnNaN(),
                                                             keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints','intermediate.hdf5'),
                                                                                             monitor='val_loss',
                                                                                             verbose=0,
                                                                                             save_best_only=False,
                                                                                             save_weights_only=False,
                                                                                             mode='auto', period=1),
                                                             keras.callbacks.TensorBoard(log_dir='./logs',
                                                                                         histogram_freq=0,
                                                                                         batch_size=64,
                                                                                         write_graph=True,
                                                                                         write_grads=False,
                                                                                         write_images=True,
                                                                                         embeddings_freq=0,
                                                                                         embeddings_layer_names=None,
                                                                                         embeddings_metadata=None)],epochs=epochs)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir, "models", model_name + '.hdf5'))
        if not os.path.exists("models"):
            os.makedirs("models")
        self.model.save(os.path.join("models",str(model_name + '.hdf5')))
        self.clean_up_logs()

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
        self.train(train_directory_ = directory_,validation_directory_ =directory_,model_name='debugging_model',epochs=10)

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

            category = category+1
