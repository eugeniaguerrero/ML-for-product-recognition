from src.DATA_PREPARATION.folder_manipulation import *

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self,dir, batch_size = 16, shuffle = True):
      'Initialization'
      print("Initialised")
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.trainx, self.trainy = self.get_files(dir)
      print(self.trainx)
      print(self.trainy)

  def generate(self):
    while 1:
        indexes = self.__get_exploration_order(self.trainx)
        # Generate batches
        imax = int(len(indexes)/self.batch_size)
        for i in range(imax):
          # Find list of IDs
          list_IDs_temp = [self.trainx[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
          y = [self.trainy[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
          # Generate data
          X = self.__data_generation(list_IDs_temp)
          print(X.shape)
          yield np.array(X), np.array(y)

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
    categories = sparsify(categories)
    return images,categories

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)
      return indexes

  def __data_generation(self,list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      #X is 4 as time is 4 steps/ images per folder
      X = np.empty((self.batch_size, 4,IM_HEIGHT, IM_WIDTH, 3))
      for i in range(len(list_IDs_temp)):
        #print(list_IDs_temp[i])
        X[i,:,:,:,:] = dstack_folder(list_IDs_temp[i])/255
        #RESCALE!!!!!!!!!
      return X

def sparsify(y):
  'Returns labels in binary NumPy array'
  n_classes = NUMBER_CLASSES# Enter number of classes
  b = np.zeros((len(y), NUMBER_CLASSES))
  b[np.arange(len(y)), y] = 1
  return b




 # Parameters
params = {'dir':'debug_folder_grouped','batch_size': 16,
                'shuffle': True}
DataGenerator(**params).generate()
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