#from src.NN_MODELS.cnn_lstm import *
from src.NN_MODELS.inception_v3 import *
from src.NN_MODELS.vgg_testing import *
#from src.NN_MODELS. import *
from src.common import *
#from src.NN_MODELS.network_tests import *
#from src.DATA_PREPARATION.partition_grouped import *
#from src.DATA_PREPARATION.partition_grouped_folders import *
import keras
from bayes_opt import BayesianOptimization
import time

lr=2
decay = 6
#########################################
######### FAILED #######################


'''
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.1,
    shear_range=0.1,
    rescale=1. / 255)
    
    datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1. / 255)
'''

datagen = ImageDataGenerator(
   rescale=1. / 255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True)
datagenval = ImageDataGenerator()

##############################################################################
################# TEST IMAGE SIZE ###########################################
############################################################################

file = open('MODEL_OUTPUTS/image_sizetime.txt',mode='w')
file.write("Image_Size,Start,End,Difference\n")

for image_size in [200]:
    start = time.time()
    vgg_ = VGG(False,IM_HEIGHT=image_size,IM_WIDTH=image_size)
    vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA, model_description= "image_test_size" + str(image_size),
               epochs=NUMBER_EPOCHS,datagen=datagen,datagenval=datagenval)
    end = time.time()
    diff = end-start
    file.write(str(image_size) + ',' + str(start) + ',' + str(end) + ',' + str(diff) + '\n')

file.close()


##############################################################################
################# OPTIMISE NETWORK ###########################################
############################################################################

#IM_HEIGHT = 150
#IM_WIDTH = 150

def get_vals(conv1_size = 5,conv2_size = 6,dense_size = 8,moment = 0.9):
    global lr
    global decay
    print("Dense Size is " + str(dense_size))
    vals = "LR-" + str(lr) + "_C1-" + str(conv1_size) + "_C2-" + str(conv2_size) + "_DS-" + str(dense_size)
    vals +=  "_D-" + str(decay) + "_M-" + str(moment)
    print(vals)
    vgg_ = VGG(output=False,lr=lr,conv1_size=conv1_size,conv2_size=conv2_size,dense_size=dense_size,decay=decay,moment=moment)
    vgg_.train(train_directory_=TRAIN_DATA, validation_directory_=VALIDATE_DATA,
               model_description="image_test_size" + str(vals), epochs=NUMBER_EPOCHS, datagen=datagen,
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
    dist = -2.1 + 4
    grid = return_grid([-6,-1.5], dist)
    for lr_ in grid:
        dist2 = 1 - 0.7
        grid2 = return_grid([0.7, 1], dist2)
        for lr_decay in grid2:
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

bo = BayesianOptimization(lambda conv1_size,conv2_size,dense_size,moment: get_vals(conv1_size,conv2_size,dense_size,moment),
                          {"conv1_size":(3,8),"conv2_size":(3,8),"dense_size":(6,11),"moment":(0.5,1.0)})

bo.explore({"conv1_size":(3,8),"conv2_size":(3,8),"dense_size":(6,11),"moment":(0.5,1.0)})
bo.maximize(init_points=2, n_iter=500, kappa=10,acq="ucb") #, acq="ucb"
