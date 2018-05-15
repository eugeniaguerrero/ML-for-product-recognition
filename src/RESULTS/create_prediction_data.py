import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pickle

n_classes = 20  # classes model was trained on (final layer)
top_layer_dimension = 250   #pixels

# Load model from file path
m_name = "C:\\Users\\GC\\Desktop\\PostExamProject\\group-project-back-end\\DATA\\11-05-17 FD 20 classes and Model\\0704_FD_ambient_data\\new_model\\intermediate.hdf5"
the_model = keras.models.load_model(m_name) # Load model from file

# Get list of class names
class_names = os.listdir("C:\\Users\\GC\\Desktop\\PostExamProject\\group-project-back-end\\DATA\\11-05-17 FD 20 classes and Model\\0704_FD_ambient_data\\FD_training_data")

# Set base path for data
base_path = 'C:\\Users\\GC\\Desktop\\PostExamProject\\group-project-back-end\\DATA\\11-05-17 FD 20 classes and Model\\0704_FD_ambient_data\\FD_training_data'

# Output files
file_object  = open("prediction_data\\training_set\\conf_matrix_validation.txt", "w")
file_object_agg  = open("prediction_data\\training_set\\conf_matrix_validation_agg.txt", "w")

# Predict all of the images for each class
# Aggregate predictions by video
for cname in class_names:
    active_path = os.path.join(base_path, cname)
    datagenval = ImageDataGenerator(rescale=1. / 255)
    validate_generator = datagenval.flow_from_directory(
        active_path,
        target_size=(top_layer_dimension, top_layer_dimension),
        batch_size=16,
        class_mode="categorical")

    # Make all predictions
    predictions = the_model.model.predict_generator(validate_generator) # Predict all the images for this class
    p_values = np.zeros(n_classes)
    # Take element with max value as the prediction
    for i in range(0, len(predictions)):
        p_values[np.argmax(predictions[i])] += 1

    # Aggregate predictions by video
    ag_p = np.zeros(shape=(validate_generator.num_classes, n_classes))
    i, j = 0, 0
    idx = validate_generator.filenames[0][0:5]
    for pa in validate_generator.filenames:  # Loop over all prediction file paths
        if (idx != validate_generator.filenames[i][0:5]):
            idx = validate_generator.filenames[i][0:5]
            j += 1
        ag_p[j] += predictions[i]
        i += 1
    # Aggregate prediction is the max of the summed individual predictions on the video
    ag_p_values = np.zeros(n_classes)
    for i in range(0, len(ag_p)):
        ag_p_values[np.argmax(ag_p[i])] += 1

    # Write both sets of predictions to file
    print(cname)
    print(p_values)
    print(ag_p_values)
    file_object.write(cname)
    file_object.write('\t')
    file_object_agg.write(cname)
    file_object_agg.write('\t')
    for k in range(0, len(p_values)):
        file_object.write(str(p_values[k]))
        file_object.write('\t')
        file_object_agg.write(str(ag_p_values[k]))
        file_object_agg.write('\t')

    file_object.write('\n')
    file_object_agg.write('\n')

    filehandler = open(os.path.join("prediction_data\\training_set", cname+"predictions.pkl"), 'wb')
    pickle.dump(predictions, filehandler)

    filehandler2 = open(os.path.join("prediction_data\\training_set", cname+"filenames.pkl"), 'wb')
    pickle.dump(validate_generator.filenames, filehandler2)

file_object.close()
file_object_agg.close()
