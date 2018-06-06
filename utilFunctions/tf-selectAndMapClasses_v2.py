import tensorflow as tf
import numpy as np
import os
import shutil 
import labelMap
import sys

# ###
# File: tf-selectAndMapClasses_v2.py
# Author: Dhruv Jain
# Functions to pre-process the dataset before ML training and evaluation.
# The main file to select videos with particular labels and map labels from original dataset to the new dataset
# Command line arguments:
# Argument1: Folder containing the dataset with original labels to be mapped
# Arugment2: Folder to output the dataset with mapped labels. 
# ### 

def selectLabels(inFolder, outFolder):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    writer = tf.python_io.TFRecordWriter(path=outFolder+'data.tfrecord')
    for infile in os.listdir(path=inFolder):
        record_iterator = tf.python_io.tf_record_iterator(path=inFolder+infile)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)
            if(len(np.intersect1d(labelMap.mapping[:,0], example.context.feature['labels'].int64_list.value))>0):
                k = len(example.context.feature['labels'].int64_list.value)
                i = 0
                # code to delete labels that are not in mapping[:,0]
                while (i < k):
                    if(not example.context.feature['labels'].int64_list.value[i] in labelMap.mapping[:,0]):
                        example.context.feature['labels'].int64_list.value.remove(example.context.feature['labels'].int64_list.value[i])
                        k=k-1
                        i=i-1
                    i=i+1
                writer.write(example.SerializeToString())
    writer.close() 
     
if os.path.exists('./temp/'):
    shutil.rmtree('./temp/')        
selectLabels(sys.argv[1], './temp/')

##
# The main function to map labels in the file
# Pseudocode
#	for each file (done)
# 		read the file (Done)
#		for each sequence in the file (Done)
#			read a sequence from the file (Done)
#			read labels in the sequence (Done)
#			if one of the labels is 355, 359 or 362 keep that sequence (Done)
#			otherwise, delete the sequence (Done)
#			delete labels that are not 355, 359, or 362
# 		if any sequence left, save the file
# 		otherwise delete the file
def mapLabelsInFile(readfile, mapping, writefile):
    record_iterator = tf.python_io.tf_record_iterator(path=readfile)
    writer = tf.python_io.TFRecordWriter(path=writefile)
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)      
        for i in range(0, len(example.context.feature['labels'].int64_list.value)):
            example.context.feature['labels'].int64_list.value[i] = labelMap.mapLabel(example.context.feature['labels'].int64_list.value[i])
        
        # Delete repeated labels if any
        unique = np.unique(example.context.feature['labels'].int64_list.value)
        i = 0
        while (i < len(unique)):
            example.context.feature['labels'].int64_list.value[i] = unique[i]
            i+=1
        while (i < len(example.context.feature['labels'].int64_list.value)):
            example.context.feature['labels'].int64_list.value.remove(example.context.feature['labels'].int64_list.value[i])
        
        writer.write(example.SerializeToString())
    writer.close()

def mapLabelsInFolder(inFolder, outFolder):
    # Reassign 354 to 387 towards 0 to 33
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    for infile in os.listdir(path=inFolder):
        mapLabelsInFile(inFolder+infile, labelMap.mapping, outFolder+infile)

if os.path.exists(sys.argv[2]):
    shutil.rmtree(sys.argv[2])  
mapLabelsInFolder('./temp/', sys.argv[2])
shutil.rmtree('./temp/')


# Function to print stats of the new dataset including the number of labels, and the number of examples per label.
def printStats(inFolder, numOfLabels):
    count = np.zeros((numOfLabels,), dtype=int)
    examples = 0
    for infile in os.listdir(path=inFolder):
        record_iterator = tf.python_io.tf_record_iterator(path=inFolder+infile)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)      
            examples+=1
            for i in range(0, len(example.context.feature['labels'].int64_list.value)):
                count[example.context.feature['labels'].int64_list.value[i]] += 1
    print ('New Dataset stats:\n' + '[[lab,#]')  
    print (np.column_stack((np.arange(0, labelMap.numOfLabels),count)))
    print ('#Lab:', numOfLabels, '#Examples:', examples, '#Overlap:', np.sum(count) - examples, 'outputFolder:', inFolder)

printStats(sys.argv[2], labelMap.numOfLabels)
