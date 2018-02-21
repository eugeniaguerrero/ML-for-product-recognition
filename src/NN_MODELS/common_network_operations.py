from src.common import *
import keras
from src.DATA_PREPARATION.folder_manipulation import *

#COMBINE ALL 3 into one??

def clean_up(model_name):
    to_clean = [TENSORBOARD_LOGS_FOLDER,JSON_LOG_FILE,MODEL_SAVE_FOLDER]
    to_make = [TENSORBOARD_OLD_LOGS_FOLDER,JSON_OLD_LOGS_FOLDER,OLD_MODELS_FOLDER]
    extension = ['','.json','.hdf5']

    for ext,folder,to_move in zip(extension,to_make,to_clean):
        if not os.path.exists(folder):
            os.makedirs(folder)
        old_list = os.listdir(folder)
        numbers = []
        for i in old_list:
            numbers.append(int(i.split('_')[0]))
        numbers = sorted(numbers)
        if len(numbers) > 0:
            count = numbers[-1] +1
        else:
            count = 0
        new_name = str(count) + '_' + model_name + ext
        os.rename(to_move, os.path.join(folder ,new_name))
        print("Data is in : " + to_move + "/" + new_name)

'''
def clean_up_logs(model_name):
    if not os.path.exists(TENSORBOARD_OLD_LOGS_FOLDER):
        os.makedirs(TENSORBOARD_OLD_LOGS_FOLDER)
    old_logs_list = os.listdir(TENSORBOARD_OLD_LOGS_FOLDER)
    numbers = []
    for i in old_logs_list:
        numbers.append(int(i.split('_')[0]))
    numbers = sorted(numbers)
    if len(numbers) > 0:
        count = numbers[-1] +1
    else:
        count = 0
    foldername = str(count) + '_' + model_name
    os.rename(TENSORBOARD_LOGS_FOLDER, os.path.join(TENSORBOARD_OLD_LOGS_FOLDER ,foldername))
    print("Tensorboard data is in : " + TENSORBOARD_OLD_LOGS_FOLDER + "/" + foldername)

def clean_up_json_logs(model_name):
    if not os.path.exists(JSON_OLD_LOGS_FOLDER):
        os.makedirs(JSON_OLD_LOGS_FOLDER)
    old_logs_list = os.listdir(JSON_OLD_LOGS_FOLDER)
    numbers = []
    for i in old_logs_list:
        numbers.append(int(i.split('_')[0]))
    numbers = sorted(numbers)
    if len(numbers) > 0:
        count = numbers[-1] +1
    else:
        count = 0
    filename = str(count) + '_' + model_name + '.json'
    os.rename(JSON_LOG_FILE, os.path.join(JSON_OLD_LOGS_FOLDER ,filename))
    print("Json data is now in : " + JSON_OLD_LOGS_FOLDER + "/" +  filename)

def clean_up_models(model_name):
    if not os.path.exists(OLD_MODELS_FOLDER):
        os.makedirs(OLD_MODELS_FOLDER)
    old_logs_list = os.listdir(OLD_MODELS_FOLDER)
    numbers = []
    for i in old_logs_list:
        numbers.append(int(i.split('_')[0]))
    numbers = sorted(numbers)
    if len(numbers) > 0:
        count = numbers[-1] +1
    else:
        count = 0
    filename = str(count) + '_' + model_name + '.hdf5'
    os.rename(os.path.join(MODEL_SAVE_FOLDER,model_name + '.hdf5'), os.path.join(OLD_MODELS_FOLDER,filename))
    print("Model data is now in : " + OLD_MODELS_FOLDER + "/" + filename)
'''


def find_incorrect_classifications(directory_, NN):
    incpred = "incorrect_predictions"
    if not os.path.exists(incpred):
        os.makedirs(incpred)

    # Test 1 check if untrained model returns uniform predictions
    folders = get_folders(directory_)
    category = 0
    import shutil
    for folder in folders:

        if not os.path.exists(os.path.join(incpred, folder)):
            os.makedirs(os.path.join(incpred, folder))

        image_list = get_image_names(os.path.join(directory_, folder))
        for image in image_list:
            filepath = os.path.join(directory_, folder, image)
            resized_image = get_image(filepath)
            predictions = NN.predict(resized_image)

            if np.argmax(predictions) != category:
                fileto = os.path.join(incpred, folder, image)
                shutil.copyfile(filepath, fileto)
        category = category + 1

def get_Tensorboard():
    tensor_log = keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_FOLDER,
                                histogram_freq=0,
                                batch_size=BATCH_SIZE,
                                write_graph=True,
                                write_grads=False,
                                write_images=True,
                                embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None)
    return tensor_log

def get_model_checkpoint():
    model_check = keras.callbacks.ModelCheckpoint(filepath=INTERMEDIATE_FILE,
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', period=1)
    return model_check