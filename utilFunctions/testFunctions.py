import numpy as np
import tensorflow as tf

def printExample(inFile):
    record_iterator = tf.python_io.tf_record_iterator(path=inFile)
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)
        print(example.context.feature['video_id'].bytes_list.value[0].decode("utf-8") == 'ZopQlxu4190')
        break
    
printExample('data.tfrecord')