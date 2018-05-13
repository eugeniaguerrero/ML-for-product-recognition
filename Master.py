#from src.NN_MODELS.cnn_lstm import *
#from src.NN_MODELS.inception_v3 import *
from src.NN_MODELS.vgg_testing import *
#from src.NN_MODELS. import *
from src.common import *
#from src.NN_MODELS.network_tests import *
#from src.DATA_PREPARATION.partition_grouped import *
#from src.DATA_PREPARATION.partition_grouped_folders import *
import keras
from bayes_opt import BayesianOptimization
import time

FD_DATA_LOC = os.path.join("DATA","ambient_data","raw_data")#"1104_raw_full_ambient")#,"1104_FD_Full_Set")
NORMAL_DATA_LOC =  os.path.join("DATA","ambient_data","raw_data")


##############################################################################
################# FRAME DIFF VS NON-FRAME DIFF ###########################################
############################################################################

IM_HEIGHT=250
FD = "True"
Preprop = "True"
conv1_size = 6
cached_model= None
moment = 0.9
decay = 6
dense_size = 8
conv2_size = 7
conv1_size = 6
lr=4
NUMBER_EPOCHS = 250
IM_WIDTH=250

file = open('MODEL_OUTPUTS/Large_Dataset_Time_to_train.txt',mode='w')
file.write("Start,End,Difference\n")
start = time.time()

for FD_ in [1]:

    if FD_ == 0:
        FD = "False"
        TRAIN_DATA = os.path.join(NORMAL_DATA_LOC,"training_data")
        VALIDATE_DATA = os.path.join(NORMAL_DATA_LOC,"validation_data")
    else:
        FD = "True"
        TRAIN_DATA = os.path.join(FD_DATA_LOC,"training_data")
        VALIDATE_DATA = os.path.join(FD_DATA_LOC,"validation_data")

    for prep_ in [0]:
        if prep_ == 0:
            Preprop = "False"
            datagen = ImageDataGenerator(rescale=1. / 255)
            datagenval = ImageDataGenerator(rescale= 1./255)
        else:
            Preprop = "True"
            datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,
                                         height_shift_range=0.1,channel_shift_range=0.1,shear_range=0.1,
                                         rotation_range = 10,rescale=1. / 255)
            datagenval = ImageDataGenerator(rescale= 1./255)

        #STRING TO SAVE MODEL AS
        vals = "LR-" + str(lr) + "_C1-" + str(conv1_size) + "_C2-" + str(conv2_size) + "_DS-" + str(dense_size)
        vals +=  "_D-" + str(decay) + "_M-" + str(moment) + "_IM-" + str(IM_WIDTH) + "_FD-" + FD + "_Time_Stamp-" + str(time.time()) + "_Preprop-" + Preprop
        #inc = INCEPTION_V3();
        #inc.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= "small_dataset" + "Preprop-" + Preprop,epochs=NUMBER_EPOCHS,datagen=datagen,datagenval=datagenval)
        vgg_ = VGG(False,IM_HEIGHT=IM_HEIGHT,IM_WIDTH=IM_WIDTH,lr=lr,conv1_size=conv1_size,conv2_size=conv2_size,dense_size=dense_size,decay=decay,moment=moment)
        vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= "full_dataset_"+ vals,epochs=NUMBER_EPOCHS,datagen=datagen,datagenval=datagenval)

end = time.time()
diff = end-start
file.write(str(start) + ',' + str(end) + ',' + str(diff) + '\n')
file.close()

##############################################################################
################# TEST IMAGE SIZE ###########################################
############################################################################

'''
#WORKS!! - 72%
#Fails over 300
datagen = ImageDataGenerator(
    horizontal_flip=True,
        vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.1,
    shear_range=0.1,
    rotation_range = 10,
    rescale=1. / 255)
'''

#just horiz flip fails shear range fails too
#datagen = ImageDataGenerator(
    #rescale=1. / 255)
'''
datagen = ImageDataGenerator(
    horizontal_flip=True,
        vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.1,
    shear_range=0.1,
    rotation_range = 10,
    rescale=1. / 255)

datagenval = ImageDataGenerator(rescale= 1./255)

NUMBER_EPOCHS = 75


for FD_ in [1]:
    if FD_ == 0:
        FD = "False"
        TRAIN_DATA = os.path.join(NORMAL_DATA_LOC,"training_data")
        VALIDATE_DATA = os.path.join(NORMAL_DATA_LOC,"validation_data")
    else:
        FD = "True"
        TRAIN_DATA = os.path.join(FD_DATA_LOC,"training_data")
        VALIDATE_DATA = os.path.join(FD_DATA_LOC,"validation_data")

    file = open('MODEL_OUTPUTS/image_sizetime.txt',mode='w')
    file.write("Image_Size,Start,End,Difference\n")
    for image_size in [50,100,150,200,250,300,350,400]:

        IM_HEIGHT=image_size
        Preprop = "True"
        conv1_size = 6
        cached_model= None
        moment = 0.9
        decay = 6
        dense_size = 8
        conv2_size = 7
        conv1_size = 6
        lr=4

        start = time.time()

        #STRING TO SAVE MODEL AS
        vals = "LR-" + str(lr) + "_C1-" + str(conv1_size) + "_C2-" + str(conv2_size) + "_DS-" + str(dense_size)
        vals +=  "_D-" + str(decay) + "_M-" + str(moment) + "_IM-" + str(image_size) + "_FD-" + FD + "_Time_Stamp-" + str(time.time()) + "_Preprop-" + Preprop

        vgg_ = VGG(False,IM_HEIGHT=IM_HEIGHT,IM_WIDTH=IM_WIDTH,lr=lr,conv1_size=conv1_size,conv2_size=conv2_size,dense_size=dense_size,decay=decay,moment=moment)
        vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= vals,
               epochs=NUMBER_EPOCHS,datagen=datagen,datagenval=datagenval)
        end = time.time()
        diff = end-start
        file.write(str(image_size) + ',' + str(start) + ',' + str(end) + ',' + str(diff) + '\n')
    file.close()
'''


##############################################################################
################# OPTIMISE NETWORK ###########################################
############################################################################

IM_HEIGHT=250
FD = "True"
Preprop = "True"
conv1_size = 6
cached_model= None
moment = 0.9
decay = 6
dense_size = 8
conv2_size = 7
conv1_size = 6
lr=4
NUMBER_EPOCHS = 50
IM_WIDTH=250


datagen = ImageDataGenerator(
    horizontal_flip=True,
        vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.1,
    shear_range=0.1,
    rotation_range = 10,
    rescale=1. / 255)

datagenval = ImageDataGenerator(rescale= 1./255)

def get_vals(conv1_size = 5,conv2_size = 6,dense_size = 8,moment = 0.9):
    global lr
    global decay
    print("Dense Size is " + str(dense_size))
    vals = "LR-" + str(lr) + "_C1-" + str(conv1_size) + "_C2-" + str(conv2_size) + "_DS-" + str(dense_size)
    vals +=  "_D-" + str(decay) + "_M-" + str(moment) + "_IM-" + str(IM_WIDTH) + "_FD-" + FD + "_Time_Stamp-" + str(time.time()) + "_Preprop-" + Preprop
    print(vals)
    TRAIN_DATA = os.path.join(FD_DATA_LOC,"training_data")
    VALIDATE_DATA = os.path.join(FD_DATA_LOC,"validation_data")
    try:
        vgg_ = VGG(False,IM_HEIGHT=IM_HEIGHT,IM_WIDTH=IM_WIDTH,lr=lr,conv1_size=conv1_size,conv2_size=conv2_size,dense_size=dense_size,decay=decay,moment=moment)
        vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA,
                   model_description="optimise" + str(vals), epochs=NUMBER_EPOCHS, datagen=datagen,
                   datagenval=datagenval)
        validate_generator = datagenval.flow_from_directory(
            VALIDATE_DATA,
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical")
        print(vgg_.model.metrics_names)
        data = vgg_.model.evaluate_generator(validate_generator)
        print(data)
        return data[1]
    except:
        return 0.0



#RETURNS A GRID BETWEEN TWO VALUES
def return_grid(value,dist):
    grid = np.arange(value[0],value[1]+dist/10,dist/10,dtype=np.float64)
    return grid

def minimise_lr_and_lr_decay():
    global kwargs
    global lr
    global decay
    global NUMBER_EPOCHS

    #NUMBER_EPOCHS = 30
    max_lr = None
    max_lr_dec = None
    max_val = None
    dist = 5
    grid = [3,3.5,4,4.5,5,5.5,6]#return_grid([3,8], dist)
    dist2 = 6
    grid2 = [0]
    #grid2 = [0.1,8]
    print("LR_GRID is " + str(grid))
    print("LR_Decay is " + str(grid2))
    for lr_decay in grid2:
        for lr_ in grid:
            values = []
            max_item = None
            max_val = 0
            decay = lr_decay
            lr = lr_
            values.append(get_vals())
            if values[-1]>max_val:
                max_val = values[-1]
                max_lr = lr_
                max_lr_dec = lr_decay
    print ("Maximum Val was : " + str(max_val) + " Lr " + str(max_lr) + " lr_dec " + str(max_lr_dec))
    NUMBER_EPOCHS = 70


minimise_lr_and_lr_decay()

'''
bo = BayesianOptimization(lambda conv1_size,conv2_size,dense_size,moment: get_vals(conv1_size,conv2_size,dense_size,moment),
                          {"conv1_size":(3,8),"conv2_size":(3,8),"dense_size":(6,11),"moment":(0.5,1.0)})

bo.explore({"conv1_size":(3,8),"conv2_size":(3,8),"dense_size":(6,11),"moment":(0.5,1.0)})
bo.maximize(init_points=2, n_iter=500, kappa=10,acq="ucb") #, acq="ucb"
'''
