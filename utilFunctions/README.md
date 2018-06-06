# Sound Classification repo by Makeability Lab.
### Author: Dhruv Jain
### This folder contains functions to pre-process the Google's audioset dataset.

labelMap.py contains parameters to specify the label selections for the current training (As a mapping between the original 527 classes to the new set of classes). This is to be used as an meta import module in other python files.

plotConfusionMatrix_UptilHit5.py plots predictions for every class and compares them against the true labels for that class (this is not to be confused with general CM which is not valid for multi-label classification). Command line argument parameter includes Hits uptil which the predictions are to be considered.

Usage: 
```bash
plotConfusionMatrix_UptilHit5.py Arg1 Arg2 Arg3 Arg4
```
Where, 
Argument 1: the infolder of dataset with true labels, 
Argument 2: inference file for predictions, 
Argument 3: the result file, 
Argument 4: the number of hits to consider uptil 5

tf-multiClassFunctions.py contains assorted functions to pre-process the dataset before ML training and evaluation.

Usage: 
```bash
tf-multiClassFunctions.py Arg1 Arg2
```

Where,
Argument 1: Infolder for dataset files, 
Argument 2: Outfolder to put the processed data

* tf-selectAndMapClasses_v2.py is the main file to select videos with particular labels and map labels from original dataset to the new dataset

Usage: 
```bash
tf-selectAndMapClasses_v2.py Arg1 Arg2
```

Where,
Argument1: Folder containing the dataset with original labels to be mapped. 
Arugment2: Folder to output the dataset with mapped labels. 
