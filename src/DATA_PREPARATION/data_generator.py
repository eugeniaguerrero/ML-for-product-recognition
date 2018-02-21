from src.DATA_PREPARATION.folder_manipulation import *
from src.PREPROCESSING.preprocessing import *
import cv2
from PIL import Image


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self,dir, batch_size = 16, shuffle = True, sequence_length = 1):
      'Initialization'
      print("Initialised")
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.trainx, self.trainy = self.get_files(dir)
      self.sequence_length_ = sequence_length
      self.steps_per_epoch_ = np.floor(len(self.trainx) / self.batch_size)

      # Initialise the pre processer
      self.preprocessor = Preprocessing(
              rotation=True,
              rotation_degrees=40,
              zoom= True,
              horizontal_flip= True,
              vertical_flip= True,
              histogram_equalisation= True)


  def generate(self):
      # For as many epochs as required:
    while 1:
        # Shuffle
        indexes = self.__get_exploration_order(self.trainx)
        # Generate batches
        n_batches = int(len(indexes)/self.batch_size)
        for i in range(n_batches):

          # Get a batch of folder names from the list of all folder names
          folder_names_slice = [self.trainx[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

          # Get the corresponding labels for the folders
          y = [self.trainy[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

          # Generate data
          X, labels = self.__data_generation(folder_names_slice, y)

        # Remove the time dimension for models that don't use it
        # Restructure so that datagen takes photos in folder as an
        # Argument instead of time steps, and a boolean flag for
        # Whether to time distribute the output by folder
          if self.sequence_length_ == 1:
              X = np.squeeze(X, axis=1)

          yield(X, labels)


    #Get lists of the image folders and labels
  def get_files(self,dir):
    folders = get_folders(dir)
    images = []
    categories = []
    cat = 0
    for folder in folders:
        image_names = get_folders(os.path.join(dir,folder))
        for i in range(len(image_names)):
            images.append(os.path.join(dir,folder,image_names[i]))
            categories.append(cat)
        cat += 1
    self.n_classes = cat
    categories = sparsify(categories)
    return images,categories

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)
      return indexes

    # Get the data and labels for a batch of folders
  def __data_generation(self, folder_names, folder_labels):

      n_pictures = FILES_IN_FOLDER
      # Initialise outputs
      # output length is the number of rows from one call to dstack
      # overall output is the length of the batch to be returned
      output_length =  n_pictures // self.sequence_length_
      total_output_length = self.batch_size * output_length
      X = np.zeros(shape =(total_output_length, self.sequence_length_, IM_WIDTH, IM_HEIGHT, NUMBER_CHANNELS))
      Y = np.zeros(shape = (total_output_length, self.n_classes))

      # Loop over the input: make call to dstack then append result with label:
      k = 0
      for i in range(self.batch_size):
          # Label
          Y[k:k+output_length] = folder_labels[i]
          # Images from folder
          x_out = dstack_folder_sequence(folder_names[i], self.sequence_length_)

            # Xout now has dimensions (batch size or time, h, w, c)
            # Should pre process then insert into X

            # Pre process batch
          # If sequence length is one then first dimension is batch size
          # Loop over the batch dimension calling pre-process with 1 in time dimension
          if self.sequence_length_ == 1:
            for j in range(x_out.shape[0]):
                #X[k + j] = self.preprocessor.preprocess_images(x_out[j])
                X[k + j] = self.preprocessor.preprocess_images(x_out[None, j]) # input shape (1, h, w ,c)

          # Else the first dimension is taken to be time
          else:
            # preproccess
            x_out = self.preprocessor.preprocess_images(x_out)
            # Add in the batch dimension
            x_out = np.expand_dims(x_out, axis=0)
            X[k:k+output_length] = x_out

          k = k + output_length


        # Do final rescaling from -1 to 1 (extend later to 0 to 1)

      #for i in range(X.shape[0]):
       # X[i] = (X[i]-127.5)/127.5

      X = (X - 127.5) / 127.5
      #X = 127.5*(X + 127.5)

      return X, Y


  # def __data_generation(self,list_IDs_temp):
  #     'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
  #     # Initialization
  #     #X is 4 as time is 4 steps/ images per folder
  #     X = np.empty((self.batch_size * self.sequence_length_ , self.sequence_length_ ,IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS))
  #     for i in range(len(list_IDs_temp)):
  #       #print(list_IDs_temp[i])
  #       pics = (dstack_folder(list_IDs_temp[i]))
  #       np.expand_dims(pics, axis=0)
  #       np.reshape(pics, (self.batch_size, self.sequence_length_, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS))
  #       #X[i,:,:,:,:] = (dstack_folder(list_IDs_temp[i])-127.5)/127.5
  #       #RESCALE!!!!!!!!!
  #     return X

def sparsify(y):
  'Returns labels in binary NumPy array'
  n_classes = NUMBER_CLASSES# Enter number of classes
  b = np.zeros((len(y), NUMBER_CLASSES))
  b[np.arange(len(y)), y] = 1
  return b



'''
 # Parameters
params = {'dir':'debug_folder_grouped','batch_size': 16,
                'shuffle': True}
DataGenerator(**params).generate()'''
'''
params = {'dir':'validation_data','batch_size': 16,
                'shuffle': True}
DataGenerator(**params).generate()'''

'''
def sparsify(y):
  'Returns labels in binary NumPy array'
  n_classes = NUMBER_CLASSES# Enter number of classes
  return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])'''