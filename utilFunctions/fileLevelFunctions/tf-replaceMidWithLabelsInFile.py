import csv
import numpy as np

midIndex = np.array(list(csv.reader(open('class_labels_indices.csv'))))  #Format: Index, Mid, Label name
segments =  np.array(list(csv.reader(open('balanced_train_segments.csv')))) #Format: Video id, start, end, mid1, mid2,... uptil mid12 

for i in range(0, segments.shape[0]):
    segments[i][0] = segments[i][0].replace('-', '')
    for j in range(3, segments.shape[1]):
        if(segments[i][j] != ''):
            segments[i][j] = segments[i][j].replace('"', '')
            segments[i][j] = segments[i][j].replace(' ', '')
            index = np.where(midIndex[:,1] == segments[i][j])[0][0]
            segments[i][j] = midIndex[index][0]
            
np.savetxt("bal_train_labels.csv", segments, fmt='%s', delimiter=',')