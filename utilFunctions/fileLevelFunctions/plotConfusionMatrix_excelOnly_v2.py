import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import labelMap
import sys
import os

# Command line arguments as input
# Argument 1:  the infolder of dataset with true labels
# Argument 2: inference file for predictions
# Argument 3: the result file 

def countExamples(inFolder):
    examples = 0
    for infile in os.listdir(path=inFolder):
        record_iterator = tf.python_io.tf_record_iterator(path=inFolder+infile)
        for string_record in record_iterator:
            examples+=1
    return examples

def readLabels(inFolder):
    true = np.ones((countExamples(inFolder), 13), dtype=int)*600    #max number of labels = 12
    true = true.astype(str)
    i = 0
    for infile in os.listdir(path=inFolder):
        record_iterator = tf.python_io.tf_record_iterator(path=inFolder+infile)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)
            true[i][0] = example.context.feature['video_id'].bytes_list.value[0].decode("utf-8")
            for j in range(0, len(example.context.feature['labels'].int64_list.value)):
                true[i][j+1] = str(example.context.feature['labels'].int64_list.value[j])
            i+=1
    return true

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    #Map original labels to label names -- only in case classes are not combined
    #labelNames = np.array(list(csv.reader(open('../metaFiles/class_labels_indices.csv')))) 
    #labelNames = np.delete(labelNames, (0), axis=0)
    #headers = np.take(labelNames[:,2], labelMap.mapping[:,0])
    #Repace all ',' in headers as it may interfer with csv formating
    #for i in range(0, headers.shape[0]):
    #    headers[i] = headers[i].replace(',', '-')
    print(cm)
    np.savetxt(sys.argv[3], cm, fmt='%s', header=np.array2string(np.arange(0, labelMap.numOfLabels), separator=',', max_line_width=9999), delimiter=',', comments='')
    
def searchLabel(array, label):
    for i in range(1, len(array)):
        if(label == int(array[i])):
            return label
    return int(array[1])

predicts = np.array(list(csv.reader(open(sys.argv[2]))))  #Format: Video id, Label prob
#predicts[1][0]                  #video id
#predicts[1][1].split(' ')[0]    #top label
true = readLabels(sys.argv[1]) #Format: Video id, label1, label2,... uptil label12
true_label = np.ones((predicts.shape[0],), dtype=int)*600
predict_label = np.ones((predicts.shape[0],), dtype=int)*600
total = 0

for i in range(1, predicts.shape[0]):
    # if (len(np.where(true[:, 0] == predicts[i][0])[0]) != 0):         # if video id in predicts[i] exists in the true video labels
    predict_label[total] = int(predicts[i][1].split(' ')[0])            # assign the top label to predict_label[total] 
    index = np.where(true[:, 0] == predicts[i][0])[0][0]                # find the relevant row in true video labels based on video id
    true_label[total] = searchLabel(true[index], predict_label[total])  # search if that predicted label exists in the original label of the video and assign accordingly. 
    total = total + 1

cm = confusion_matrix(true_label[:total], predict_label[:total], np.arange(0, labelMap.numOfLabels))
plot_confusion_matrix(cm, classes=np.arange(0, labelMap.numOfLabels), title='Confusion matrix (without normalization)')