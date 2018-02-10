from src.DATA_PREPARATION.folder_manipulation import *
from keras.utils import to_categorical

class DataGenerator(object):
    def __init__(self,dir, batch_size = 16, shuffle = True):
        print("Initialised")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.trainx, self.trainy = self.get_files(dir)
        self.data_length = len(self.trainx)

    def generate(self):
        while 1:
            indexes = self.__get_exploration_order(self.trainx)
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [self.trainx[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                y = [self.trainy[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                #y = self.sparsify(y)
                X = self.__data_generation(list_IDs_temp)
                yield np.array(X), np.array(y)

    def get_files(self,dir):
        folders = get_folders(dir)
        images = []
        categories = []
        cat = 0
        for folder in folders:
            image_names = get_image_names(os.path.join(dir,folder))
            for i in range(len(image_names)):
                images.append(os.path.join(dir,folder,image_names[i]))
                categories.append(cat)
            cat += 1

        return images,categories

    def get_batches_per_epoch(self):
        return round(self.data_length/self.batch_size)

    def __get_exploration_order(self, list_IDs):
          indexes = np.arange(len(list_IDs))
          if self.shuffle == True:
              np.random.shuffle(indexes)

              return indexes

    def __data_generation(self,list_IDs_temp):
        X = np.empty((self.batch_size, IM_HEIGHT, IM_WIDTH, 3))
        for i in range(len(list_IDs_temp)):
            X[i,:,:,:] = (get_image(list_IDs_temp[i])-127.5)/127.5
        return X

    def sparsify(self,y):
        categories = np.zeros((len(y),NUMBER_CLASSES))
        for i in range(len(y)):
            categories[i,:] = to_categorical(y[i],NUMBER_CLASSES)
        return categories
