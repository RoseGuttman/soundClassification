import numpy as np
import os
import tensorflow as tf
import sys
import shutil

## 
# File: selectReratedVideos.py
# Author: Dhruv Jain
# File to select only the rerated Videos from the audioset. Read more in the quality assessment documentation: https://research.google.com/audioset/download.html
#Argument 1: Infolder containing all video ids
#Arugment 2: Outfolder to output only rerated videos
#Arugment 3: metafile for rerated video ids
# Example
#../audioset_v1_embeddings/bal_train/
#../audioset_v1_embeddings/bal_train_rerated/
#rerated_video_ids.txt

def selectVideos(inFolder, outFolder):
    reVidId = np.loadtxt(sys.argv[3], dtype=str)
    #print ('ZZZxSkSh0Cw' in reVidId)
    writer = tf.python_io.TFRecordWriter(path=outFolder+'reratedVideos.tfrecord')
    for infile in os.listdir(path=inFolder):
        record_iterator = tf.python_io.tf_record_iterator(path=inFolder+infile)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)
            if(example.context.feature['video_id'].bytes_list.value[0].decode("utf-8") in reVidId):
                writer.write(example.SerializeToString())
    writer.close() 

if os.path.exists(sys.argv[2]):
    shutil.rmtree(sys.argv[2]) 
os.makedirs(sys.argv[2])
selectVideos(sys.argv[1], sys.argv[2])

