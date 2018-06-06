# Sound Classification repo by Makeability Lab.
# Author: Dhruv Jain and Shuxu Tian

Repo for ongoing sound classification research at UW Makeability Lab baesd on Google's Audioset dataset.

utilFunctions: Functions to down sample, select labels and do other pre processing on the Audioset. 

youtube-8m: Custom modified code for training on the audioset. Includes three NN models modified from original youtube-8m starter code. 

realTimeClassifier: Contains code to perform classification in real-time. Trained model should be added in the realTimeClassifier/model folder. Specify parameters in audio/params.py

* [Google AudioSet](https://research.google.com/audioset/)
* [Original YouTube-8M model](https://github.com/google/youtube-8m)
* [Tensorflow vggish model](https://github.com/tensorflow/models/tree/master/research/audioset)
