# Code to list all the labels in a folder containing tf-record sequence file audio embeddings for Google's audioset dataset (for format see: https://research.google.com/audioset/download.html) 

import tensorflow as tf
import numpy as np
import os

def checkLabelsinFile(readfile, listLabels):
    record_iterator = tf.python_io.tf_record_iterator(path=readfile)
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)        
        listLabels = np.union1d(listLabels, example.context.feature['labels'].int64_list.value)     
    return listLabels

def checkFolder(inFolder):
    listLabels = []
    for infile in os.listdir(path=inFolder):
        listLabels = checkLabelsinFile(inFolder+infile, listLabels)
    print(listLabels)      
    
checkFolder('./eval_8/')
