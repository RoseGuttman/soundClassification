import tensorflow as tf
import numpy as np
import os

reqLabels = list(range(354,388)) # All Domenstic, Home sounds 354-387 

# read the file
def modifyTfRecord(readfile, writefile):
    record_iterator = tf.python_io.tf_record_iterator(path=readfile)
    writer = None
    isfileCreated = False
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)
        #print(example.context.feature['labels'].int64_list.value) 
        #print ("\n")
        if(len(np.intersect1d(reqLabels, example.context.feature['labels'].int64_list.value))>0):
            if(not isfileCreated):
                isfileCreated = True
                writer = tf.python_io.TFRecordWriter(path=writefile)
            k = len(example.context.feature['labels'].int64_list.value)
            i = 0
            while (i < k):
                if(not example.context.feature['labels'].int64_list.value[i] in reqLabels):
                    example.context.feature['labels'].int64_list.value.remove(example.context.feature['labels'].int64_list.value[i])
                    k=k-1
                    i=i-1
                i=i+1
            writer.write(example.SerializeToString())
        # code to delete labels that are not in reqLabels
    if (isfileCreated):
        writer.close() 
         
def modifyTfFolder(inFolder, outFolder):
    for infile in os.listdir(path=inFolder):
        modifyTfRecord(inFolder+infile, outFolder+infile)
        
    
modifyTfFolder('./bal_train/', './output/')


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


## SANDBOX

# To know the structure of the file
#for example in tf.python_io.tf_record_iterator("test.tfrecord"):
#    structure = tf.train.Example.FromString(example)
 # this, however, does not work:
        #audio_embedding = (example.features.feature['audio_embedding'].bytes_list.value[0])

