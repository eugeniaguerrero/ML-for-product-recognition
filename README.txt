Summary:

This library contains functions to contrast and compare different
network architectures with additional features such as Frame-Differencing,
histogram equalisation and the use of built-in Keras preprocessing libraries.

To Run:

Install relevant packages from the requirements.txt found in the root directory.

Change paremeters in the file common.py to point to the data, determine the number of classes,
image size etc that are required for the data. Change some paremeters and run the Master.py file in the root directory,
 this will train all models and create relevant output files. Other functionalities such as bayesian optimisation,
 image size comparison etc. can be toggled in the Master file also.



Folder layout:

Main (Conatains this file)
DATA          >
              ALL DATA HERE
DATA          >
---------------
MODEL_OUTPUTS >
              checkpoints(folder-created by program contains saved model)
              old_json (Created by program contains old runs of model)
              old_logs (Created by program contains older run's tensorboard logs)
              old_models (Final models of a run)
MODEL_OUTPUTS >
---------------
src           >
              DATA_PREPARATION  >
                                data_generator.py (used for CNN gets images to feed in and preprocesses)
                                folder_manipulation.py (contains basic file and folder manipulation functions)
                                partition_grouped_folders.py (moves images into folders depending on time_stamp)
              DATA_PREPARATION  >
              -------------------
              NN_MODELS         >
                                cnn_lstm_ourdatagen.py (CNN model)
                                common_network_operations.py (Basic NN model operations common with all models)
                                inception_v3.py (INCEPTION V3 model)
                                vgg_net.py (VGG-like convnet network)
                                vgg_testing.py (VGG-like convnet altered for optimisation)
              NN_MODELS         >
              -------------------
              PREPROCESSING     >
                                frame_differencing_folders.py
                                histogram_equalisation.py
                                preprocessing.py
                                rotation_zoom_flip.py
              PREPROCESSING     >
              -------------------
              unittests         >
                                test_datagenerator.py (test the custom-made datagenerator)
                                test_network.py (test the networks)
                                test_preprocessing.py
              unittests         >
              -------------------
              callbacks.py (Callbacks to Slack and Json)
              common.py (Contains common parameters for the code)
src           >
--------------
Master.py (contains the main running code including optimisation and testing)
