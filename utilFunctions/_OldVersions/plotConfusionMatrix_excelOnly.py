import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import labelMap
import sys

# Command line arguments as input
# Argument 1: metafile for true labels
# Argument 2: inference file for predictions
# Argument 3: the result file 

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    #Map original labels to label names
    labelNames = np.array(list(csv.reader(open('../metaFiles/class_labels_indices.csv')))) 
    labelNames = np.delete(labelNames, (0), axis=0)
    headers = np.take(labelNames[:,2], labelMap.mapping[:,0])
    
    #Repace all ',' in headers as it may interfer with csv formating
    for i in range(0, headers.shape[0]):
        headers[i] = headers[i].replace(',', '-')
    
    print(cm)
    np.savetxt(sys.argv[3], cm, fmt='%s', header=np.array2string(headers, separator=',', max_line_width=9999), delimiter=',', comments='')
    
def searchLabel(array, label):
    for i in range(3, len(array)):
        if(array[i] == ''):
            return int(array[3])
        elif(label == int(array[i])):
            return label
    return int(array[3])

predicts = np.array(list(csv.reader(open(sys.argv[2]))))  #Format: Video id, Label prob
predicts[1][0]                  #video id
predicts[1][1].split(' ')[0]    #top label
true = np.array(list(csv.reader(open(sys.argv[1])))) #Format: Video id, start, end, label1, label2,... uptil label12
true_label = np.ones((predicts.shape[0],), dtype=int)*600
predict_label = np.ones((predicts.shape[0],), dtype=int)*600
total = 0

for i in range(1, predicts.shape[0]):
    if (len(np.where(true[:, 0] == predicts[i][0])[0]) != 0):               # if video id in predicts[i] exists in the  original video labels
        predict_label[total] = labelMap.getOrigLabel(int(predicts[i][1].split(' ')[0]))   # assign the top label to predict_label[total] based on the original label
        index = np.where(true[:, 0] == predicts[i][0])[0][0]                # find the relevant row in original video labels based on video id
        true_label[total] = searchLabel(true[index], predict_label[total])  # search if that predicted label exists in the original label of the video and assign accordingly. 
        total = total + 1

cm = confusion_matrix(true_label[:total], predict_label[:total], labelMap.mapping[:, 0])
plot_confusion_matrix(cm, classes=labelMap.mapping[:, 0], title='Confusion matrix (without normalization)')