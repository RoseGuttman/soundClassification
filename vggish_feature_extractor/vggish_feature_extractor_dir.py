# Copyright 2018 Author: Dhruv Jain, modified from vggish tensorflow model code.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Sample code to convert audio files in external datasets to the same format as embedding features 
released in AudioSet. Combines multiple blocks (feature extraction, model definition and loading, 
postprocessing) work.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are  written in a 
SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

The context(labels and video id) are inferered from the file name (dataset dependent).
                                         
Usage:
  # Run a WAV file through the model and also write the embeddings to a TFRecord file.
  # Optionally, the model checkpoint and pca_params file can be passed as argument.
  $ python vggish_inference_demo_dir.py --wav_dir ./path/to/directory/with/wave/files 
                                        --tfrecord_file /path/to/output/tensorflow/record/file
                                        --dataset dataset Name
"""

from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os

flags = tf.app.flags

flags.DEFINE_string(
    'wav_dir', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

flags.DEFINE_string(
    'dataset', None, 
     'Name of the dataset collection for audio files.')

FLAGS = flags.FLAGS


def main(_):  
  # If needed, prepare a record writer to store the postprocessed embeddings.
  writer = tf.python_io.TFRecordWriter(
    FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None
  
  # In this simple example, we run the examples from all audio files in a directory through the model.
  for infile in os.listdir(path=FLAGS.wav_dir):
    wav_file = str(FLAGS.wav_dir) + infile
    
    #####Parse file name for the video id  depending on the dataset ####
    
    ## EC 50 Dataset, Format: cross validation group - sound file id from freesound - Sound segment - Label (0 to 49).wav ##
    if(FLAGS.dataset == "EC50"):
        namegps = infile.split('-')    
        videoid = namegps[0]+"-"+namegps[1]+"-"+namegps[2] 
        label = namegps[3].split('.')[0]
    
    ## Urban Sound Dataset, Format: sound file id from freesound - Label (0 to 9) - occurance id - sound segment id.wav ##
    elif(FLAGS.dataset == "UrbanSound"):
        namegps = infile.split('-')
        videoid = namegps[0]+"-"+namegps[2]+"-"+namegps[3].split('.')[0]
        label = namegps[1]
    
    else:
        print("Please specify one of the supported datasets.")
        
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    print(examples_batch)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        print(postprocessed_batch)

        # Write the postprocessed embeddings as a SequenceExample, in a similar
        # format as the features released in AudioSet. Each row of the batch of
        # embeddings corresponds to roughly a second of audio (96 10ms frames), and
        # the rows are written as a sequence of bytes-valued features, where each
        # feature value contains the 128 bytes of the whitened quantized embedding.
        seq_example = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'labels':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(label)])),
                    'video_id':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(videoid, 'utf-8')]))
                }
            ),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                        tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[embedding.tobytes()]))
                                for embedding in postprocessed_batch
                            ]
                        )
                }
            )
        )
        print(seq_example)
        if writer:
          writer.write(seq_example.SerializeToString())

  if writer:
    writer.close()

if __name__ == '__main__':
  tf.app.run()

def printStats(inFolder, numOfLabels):
    count = np.zeros((numOfLabels,), dtype=int)
    examples = 0
