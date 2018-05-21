import tensorflow as tf
import numpy as np
import os

def reassignInFile(readfile, subNum, writefile):
    record_iterator = tf.python_io.tf_record_iterator(path=readfile)
    writer = tf.python_io.TFRecordWriter(path=writefile)
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)      
        for i in range(0, len(example.context.feature['labels'].int64_list.value)):
            example.context.feature['labels'].int64_list.value[i] = example.context.feature['labels'].int64_list.value[i] - subNum
        writer.write(example.SerializeToString())
    writer.close()

def reassignInFolder(inFolder, outFolder):
    # Reassign 354 to 387 towards 0 to 33
    subNum = 354
    for infile in os.listdir(path=inFolder):
        listLabels = reassignInFile(inFolder+infile, subNum, outFolder+infile)
    print(listLabels)      
    
reassignInFolder('./output/', './output_reassigned/')
