from src.common import *

#COMBINE ALL 3 into one??

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