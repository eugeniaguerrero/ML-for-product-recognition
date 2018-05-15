from confusion_matrix import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import os

# Read in confusion matrix file
data = pd.read_csv("prediction_data\\validation_set\\conf_matrix_validation.txt", delimiter='\t', header=None, usecols=range(0,21))
conf_matrix_df = data.as_matrix()
data = conf_matrix_df[:,1:]

# Unpack table into prediction and label vectors to make sklearn conf maxtrix
i, j = 0,0
labels = []
predictions = []
for i in range(0,20):
    for j in range(0,20):
        for k in range(0,int(data[i,j])):
            labels.append(i)
            predictions.append(j)

labs = np.array(labels)
preds = np.array(predictions)
cf = confusion_matrix(labs, preds)

class_names = os.listdir("C:\\Users\\GC\\Desktop\\PostExamProject\\group-project-back-end\\DATA\\11-05-17 FD 20 classes and Model\\0704_FD_ambient_data\\FD_training_data")

'''
RESULTS
'''
#Evaluate metrics
acc = accuracy_score(labs, preds)
prec = precision_score(labs, preds, average=None)
rec = recall_score(labs, preds, average=None)
f1 = f1_score(labs, preds,average=None)
rep = classification_report(labs, preds)

tf = open("report.txt", "w")
tf.write(rep)
tf.close()

# Plot confusion matrix
conf_plot = plot_confusion_matrix(cf, classes=class_names, normalize= False)
conf_plot.savefig("cfplot.png")
# Create boxplots
plt.boxplot([prec, rec, f1])
plt.xticks([1, 2, 3], ['Precision', 'Recall', 'F1'], size = 17)
plt.yticks(size = 17)