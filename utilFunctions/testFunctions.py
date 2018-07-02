# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:17:59 2018

@author: black
"""
import numpy as np
import tensorflow as tf

# Function to read and display a tfrecord
def readTfRecord(filename):
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)
        print (example)

readTfRecord('catAndDog.tfrecord')