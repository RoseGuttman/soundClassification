import tensorflow as tf
import numpy as np
import os
import shutil 
import labelMap

def selectLabelsInTfRecord(readfile, writefile):
    record_iterator = tf.python_io.tf_record_iterator(path=readfile)
    writer = None
    isfileCreated = False
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)
        #print(example.context.feature['labels'].int64_list.value) 
        #print ("\n")
        if(len(np.intersect1d(labelMap.mapping[:,0], example.context.feature['labels'].int64_list.value))>0):
            if(not isfileCreated):
                isfileCreated = True
                writer = tf.python_io.TFRecordWriter(path=writefile)
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
    if (isfileCreated):
        writer.close() 
         
def selectLabelsInFolder(inFolder, outFolder):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    for infile in os.listdir(path=inFolder):
        selectLabelsInTfRecord(inFolder+infile, outFolder+infile)
     
if os.path.exists('./temp/'):
    shutil.rmtree('./temp/')        
selectLabelsInFolder(labelMap.inFolder, './temp/')

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
        writer.write(example.SerializeToString())
    writer.close()

def mapLabelsInFolder(inFolder, outFolder):
    # Reassign 354 to 387 towards 0 to 33
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    for infile in os.listdir(path=inFolder):
        mapLabelsInFile(inFolder+infile, labelMap.mapping, outFolder+infile)

mapLabelsInFolder('./temp/', labelMap.outFolder)
shutil.rmtree('./temp/')

def printStats(inFolder, numOfLabels):
    count = np.zeros((numOfLabels,), dtype=int)
    for infile in os.listdir(path=inFolder):
        record_iterator = tf.python_io.tf_record_iterator(path=inFolder+infile)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)      
            for i in range(0, len(example.context.feature['labels'].int64_list.value)):
                count[example.context.feature['labels'].int64_list.value[i]] += 1
    print ('New Dataset stats:\n' + '[[lab,#]')  
    print (np.column_stack((labelMap.mapping[:, 1],count)))
    print ('#Lab:', numOfLabels, '#Examples:', np.sum(count), 'outputFolder:', inFolder)

printStats(labelMap.outFolder, labelMap.numOfLabels)
                