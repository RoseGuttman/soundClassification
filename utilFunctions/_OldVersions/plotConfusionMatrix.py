import csv
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import labelMap

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    np.savetxt('confusion_matrix.csv', cm, fmt='%s', header=np.array2string(labels, separator=',', max_line_width=9999), delimiter=',')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def searchLabel(array, label):
    for i in range(3, len(array)):
        if(array[i] == ''):
            return int(array[3])
        elif(label == int(array[i])):
            return label
    return int(array[3])

predicts = np.array(list(csv.reader(open(labelMap.inferenceFile))))  #Format: Video id, Label prob
predicts[1][0]                  #video id
predicts[1][1].split(' ')[0]    #top label
true = np.array(list(csv.reader(open(labelMap.metaFile)))) #Format: Video id, start, end, label1, label2,... uptil label12
#print(true[0][3])               #label 1 
#print (len(np.where(true[:, 1] == predicts[2][0])[0]) != 0)
true_label = np.ones((predicts.shape[0],), dtype=int)*600
predict_label = np.ones((predicts.shape[0],), dtype=int)*600
total = 0

for i in range(1, predicts.shape[0]):
    if (len(np.where(true[:, 0] == predicts[i][0])[0]) != 0):               # if video id in predicts[i] exists in the  original video labels
        predict_label[total] = labelMap.getOrigLabel(int(predicts[i][1].split(' ')[0]))   # assign the top label to predict_label[total] based on the original label
        index = np.where(true[:, 0] == predicts[i][0])[0][0]                # find the relevant row in original video labels based on video id
        true_label[total] = searchLabel(true[index], predict_label[total])  # search if that predicted label exists in the original label of the video and assign accordingly. 
        total = total + 1

cm = confusion_matrix(true_label[:total], predict_label[:total], labels)
plt.figure(figsize=(15, 15))
plot_confusion_matrix(cm, classes=labels, title='Confusion matrix (without normalization)')
plt.savefig('confusion_matrix.png')
#plt.figure(figsize=(15, 15))
#plt.savefig('confusion_matrix_nm.png')
#plot_confusion_matrix(cm, classes=labels, normalize=True, title='Confusion matrix (with normalization)')
plt.show()
