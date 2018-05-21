import tensorflow as tf
import numpy as np
import os
import csv

qualityFile = np.array(list(csv.reader(open("C:/Users/black/Google Drive/1.Research/project folders/GlassEar/ProjectSoundClassification/metafiles/qualityEstimates_ForAllRerataedClasses.csv"))))
qualityFile = np.delete(qualityFile, (0), axis=0)       # Delete file header
labelIndex = np.zeros((527), dtype=int)
total = 0
#criteria = '100' #only perfect labels
criteria = 'Yes'
for i in range(0, qualityFile.shape[0]):
    if(qualityFile[i][7] == criteria):
        labelIndex[total] = int(qualityFile[i][0])
        total += 1
print (repr(labelIndex[:total]))
