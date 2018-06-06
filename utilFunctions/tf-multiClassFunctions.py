import tensorflow as tf
import numpy as np
import os
import shutil 
import labelMap
import sys

# ###
# File: tf-multiClassFunctions.py
# Author: Dhruv Jain 
# Functions to pre-process the dataset before ML training and evaluation.
# 
# Command line arguments:
# Argument 1: Infolder for dataset files
# Argument 2: Outfolder to put the processed data
###

def countStats(inFile):
    examples = 0
    count = np.zeros((labelMap.numOfLabels,), dtype=int)
    record_iterator = tf.python_io.tf_record_iterator(path=inFile)
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)
        examples+=1
        for i in range(0, len(example.context.feature['labels'].int64_list.value)):
            count[example.context.feature['labels'].int64_list.value[i]] += 1
    return (count, examples)


##
# Function to undersample a single majority class when the labels are unique
# psuedocode:
#  identify the majority class by counting labels
#  identify the second majority class by counting labels
#  randomly remove samples from the majority class until the sample equal the second majority class or there are no more unique labels for that class.
#  export
## 
def unSampMajority(inFolder, numOfLabels, writefile):
    count = np.zeros((numOfLabels,), dtype=int)
    examples = 0
    for infile in os.listdir(path=inFolder):
        (c, e) = countStats(inFolder+infile)
        examples+=e
        count = np.add(count, c)
    majLab=np.argmax(count)
    maj2Lab = np.argsort(count)[len(count)-2]    
    #ERROR: Number of unique labels of majority class not all labels
    #To fix: do this randomly
    #np.random.choice(count[majIndex], count[majIndex] - count[maj2Index], replace=False) # generate random number array to sample from majority class 
    
    writer = tf.python_io.TFRecordWriter(path=writefile)
    for infile in os.listdir(path=inFolder):
        removingDone = False
        record_iterator = tf.python_io.tf_record_iterator(path=inFolder+infile)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)                 
            if(len(example.context.feature['labels'].int64_list.value) != 1 or example.context.feature['labels'].int64_list.value[0] != majLab or removingDone):      #if the majority label does not exit or majority label is not unique for this example, then write it.      
                writer.write(example.SerializeToString())                   
            else:
                count[majLab]-=1
                examples-=1
                if(count[majLab] <= count[maj2Lab]):
                    removingDone = True
    writer.close()
    print ('Undersampled Dataset stats:\n' + '[[lab,#]')  
    print (np.column_stack((np.arange(0, labelMap.numOfLabels),count)))
    print ('#Lab:', numOfLabels, '#Examples:', examples, '#Overlap:', np.sum(count) - examples, 'outputFile:', writefile)

if os.path.exists(sys.argv[2]):
    shutil.rmtree(sys.argv[2])
os.makedirs(sys.argv[2])
unSampMajority(sys.argv[1], labelMap.numOfLabels, sys.argv[2]+'unSampData.tfrecord')

child = np.array([ 1,  2,  3,   4,   5,  7,  14, 19, 21, 22])
parent = np.array([0,  0,  0,   0,   0,  6,  13, 18, 20, 20])

## 
# Function to attempt to partially correct the dataset: assign parent label class to all the child labels (e.g. child speech should also have speech)
# Assumption: The parent child class have to be manually put in from the ontology for all the classes I select in the new label format
## 
def assignParentLabels(readfile, writefile):
    record_iterator = tf.python_io.tf_record_iterator(path=readfile)
    writer = tf.python_io.TFRecordWriter(path=writefile)
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)      
        intersect = np.intersect1d(example.context.feature['labels'].int64_list.value, child)   #Find child elements in the given example
        
        for i in range(0, len(intersect)):
            addLabel = parent[np.where(child==intersect[i])[0][0]]
            example.context.feature['labels'].int64_list.value.append(addLabel)                 #Append parent elements in the given example
        
        # Delete repeated labels if any
        unique = np.unique(example.context.feature['labels'].int64_list.value)
        i = 0
        while (i < len(unique)):
            example.context.feature['labels'].int64_list.value[i] = unique[i]
            i+=1
        while (i < len(example.context.feature['labels'].int64_list.value)):
            example.context.feature['labels'].int64_list.value.remove(example.context.feature['labels'].int64_list.value[i])
        
        # Sort Labels
        sort = np.sort(example.context.feature['labels'].int64_list.value)
        for i in range(0, len(example.context.feature['labels'].int64_list.value)):
            example.context.feature['labels'].int64_list.value[i] = sort[i]
        
        writer.write(example.SerializeToString())
    writer.close()
    
    (count, examples) = countStats(writefile)
    print ('Parent Labels added, stats:\n' + '[[lab,#]')  
    print (np.column_stack((np.arange(0, labelMap.numOfLabels),count)))
    print ('#Lab:', labelMap.numOfLabels, '#Examples:', examples, '#Overlap:', np.sum(count) - examples, 'outputFile:', writefile)

    
#assignParentLabels(sys.argv[2]+'unSampData.tfrecord', sys.argv[2]+'unSampDataParentLabel.tfrecord')
#os.remove(sys.argv[2]+'unSampData.tfrecord')
