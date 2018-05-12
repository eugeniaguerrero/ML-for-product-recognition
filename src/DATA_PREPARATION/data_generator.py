from src.DATA_PREPARATION.folder_manipulation import *
from src.PREPROCESSING.preprocessing import *
from keras.utils import to_categorical
from PIL import Image

# Class to yield batches of image data indefinitely for as many epochs as required
class DataGenerator(object):
    def __init__(self,dir, batch_size = 16, shuffle = True, sequence_length = 1, time_distributed = False, debug_mode = False):
        print("New instance of 'datagenerator' initialised on {}".format(dir))
        # Attributes initialised directly from arguments
        self.n_classes = NUMBER_CLASSES
        self.time_distributed_ = time_distributed
        self.debug_mode_ = debug_mode

        self.shuffle = shuffle
        self.sequence_length_ = sequence_length
        self.epoch_number = 1

        # Initialise preprocessor
        self.preprocessor = Preprocessing(
            rotation=True,
            rotation_degrees=30,
            zoom=True,
            horizontal_flip=True,
            vertical_flip=True,
            histogram_equalisation=True)

        # Produce lists of all the image sequence files in the folders (trainx)
        # and corresponding one-hot array labels (trainy)
        self.trainx, self.trainy = self.get_files(dir)

        # Batch size is defined here as the size of the first dimension of the output
        # If not time distributed this is the number of images in the batch
        # If time distributed there will be batch_size*sequence_length images per batch
        self.batch_size = batch_size

        self.effective_batch_size = batch_size
        if not self.time_distributed_:
            self.effective_batch_size = int(np.floor(batch_size/self.sequence_length_))

        # Set the number of batches per epoch to iterate over in generation
        if time_distributed:
            self.batches_per_epoch =  int(np.floor(len(self.trainx) / self.batch_size))
        else:
            self.batches_per_epoch = int(np.floor(len(self.trainx) * self.sequence_length_ / self.batch_size))


    def generate(self):
        # For as many epochs as required:
        while 1:
            indexes = self.__get_exploration_order(self.trainx)
            # Generate batches
            for i in range(self.batches_per_epoch):
                folder_names_slice = [self.trainx[k] for k in indexes[i*self.effective_batch_size:(i+1)*self.effective_batch_size]]
                # Get the corresponding labels for the folders
                y = [self.trainy[k] for k in indexes[i*self.effective_batch_size:(i+1)*self.effective_batch_size]]
                # Generate data
                X, labels = self.__data_generation(folder_names_slice, y)

                # Remove the time dimension for models that don't use it
                if not self.time_distributed_:
                    X = np.squeeze(X, axis=1)

                yield(X, labels)
            self.epoch_number+=1


    #Get lists of the image folders and labels
    def get_files(self,dir):
        folders = get_folders(dir)
        images = []
        categories = []
        cat = 0
        for folder in folders:
            sequence_names = get_folders(os.path.join(dir,folder))
            for i in range(len(sequence_names)):
                images.append(os.path.join(dir,folder,sequence_names[i]))
                categories.append(cat)
            cat += 1
        categories = to_categorical(categories, num_classes=self.n_classes)
        return images,categories

    def __get_exploration_order(self, list_IDs):
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    # Get the data and labels for a batch of folders
    def __data_generation(self, folder_names, folder_labels):
        # Initialise outputs
        #  output length is the number of rows in the output from one folder
        output_length = 1
        if not self.time_distributed_:
            output_length = self.sequence_length_

        X = None
        if self.time_distributed_:
            X = np.zeros(shape=(self.batch_size, self.sequence_length_, IM_WIDTH, IM_HEIGHT, NUMBER_CHANNELS))
        else:
            X = np.zeros(shape=(self.batch_size, 1, IM_WIDTH, IM_HEIGHT, NUMBER_CHANNELS))
        Y = np.zeros(shape=(self.batch_size, self.n_classes))

        # Loop over the input: make call to dstack then append result with label:
        k = 0
        for i in range(self.effective_batch_size):
            # Get labels
            Y[k:k+output_length] = folder_labels[i]

            # Get images
            x_out = dstack_folder_sequence(folder_names[i], self.sequence_length_)
            print(folder_names[i])

            #  Pre process as time distributed set
            if self.time_distributed_:
                x_out = self.preprocessor.preprocess_images(x_out)
                X[k:k+output_length] = x_out

            # Pre process non time_distributed
            else:
                for j in range(x_out.shape[0]):
                    print(k)
                    X[k + j] = self.preprocessor.preprocess_images(x_out[None, j])  # input shape (1, h, w ,c)

            k = k + output_length

        # Apply scaling. Move to preprocessing!
        X = (X - 127.5) / 127.5
        return X, Y
