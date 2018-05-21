import numpy as np

### Iteration 1.0. All Domestic, Home sounds 354-387 ###
#numOfLabels = 34 #Total number of labels selected from the original set
#mapping = np.zeros((numOfLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
#mapping[:, 0] = np.arange(354,388) #Specify all labels to be selected from original 527 classes
#mapping[:, 1] = np.arange(0, numOfLabels) #Specify the mapping in the new label set

### Iteration 1.1. 19 Selected Domestic, Home Sounds, mapping specified in ~./Google Drive/confusion_matrix/confusion_matrix_bal_train_Logistic-eg25k-iter500-batch1k-classes34.xlsx ###
#numOfLabels = 29 #Total number of labels selected in the original set
#mapping = np.zeros((numOfLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
#mapping[:, 0] = np.array([354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 382, 383, 384, 385, 386])  #Specify all labels to be selected from original 527 classes
#mapping[:, 1] = np.array([0,   1,   1,   0,   2,   3,   4,   5,   0,   0,   6,   7,   8,   9,   10,  11,  12,  13,  13,  14,  12,  15,  16,  14,  16,  17,  18,  18,  18]) 

### Iteration 1.2. 7 Selected Speech sounds ###
#numOfLabels = 7 #Total number of labels selected from the original set of 537 classes
#mapping = np.zeros((numOfLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
#mapping[:, 0] = np.array([0, 1, 2, 3, 4, 5, 6])  #Specify all labels to be selected from original 527 classes
#mapping[:, 1] = np.array([0, 1, 2, 3, 4, 5, 6]) 

### Iteration 1.3. 10 Selected Home sounds mapping specified in ~./GoogleDrive/qualityEstimates.xlsx###
#numOfLabels = 10 #Total number of labels selected from the original set of 537 classes
#mapping = np.zeros((numOfLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
#mapping[:, 0] = np.array([367, 368, 369, 370, 371, 374, 377, 382, 384, 386])  #Specify all labels to be selected from original 527 classes
#mapping[:, 1] = np.arange(0, numOfLabels)

### Iteration 1.4. 8 Selected Home sounds with 90% or more accuracy. mapping specified in ~./GoogleDrive/qualityEstimates.xlsx###
#numOfLabels = 8 #Total number of labels selected from the original set of 537 classes
#mapping = np.zeros((numOfLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
#mapping[:, 0] = np.array([361, 367, 368, 370, 371, 374, 377, 386])  #Specify all labels to be selected from original 527 classes
#mapping[:, 1] = np.arange(0, numOfLabels)
#inFolder='./bal_train/'
#outFolder='./bal_train_8/'


## Iteration 1.5. 79 Selected classes from all 527 classes with 100% accuracy. mapping specified in ~./GoogleDrive/qualityEstimates.xlsx###
#numOfLabels = 79 #Total number of labels selected from the original set of 537 classes
#mapping = np.zeros((numOfLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
#mapping[:, 0] = np.array([  0,   2,   3,   5,  16,  18,  23,  27,  36,  43,  47,  49,  58,
#        59,  60,  66,  72,  74,  75,  81,  94,  96,  98, 101, 111, 112,
#       113, 127, 131, 137, 138, 139, 143, 162, 164, 165, 168, 170, 172,
#       174, 191, 194, 195, 200, 283, 288, 292, 300, 306, 307, 322, 327,
#       328, 331, 332, 336, 339, 343, 347, 361, 367, 371, 386, 396, 397,
#       407, 418, 421, 426, 427, 432, 444, 454, 458, 468, 481, 500, 510, 525])  #Specify all labels to be selected from original 527 classes
#mapping[:, 1] = np.arange(0, numOfLabels)
#inFolder='../audioset_v1_embeddings/bal_train/'
#outFolder='../audioset_v1_embeddings/bal_train_79/'
#iterationName='Logistic-eg100k-iter100-batch512-classes79'
#metaFile='./metaFiles/bal_train_labels.csv'

## Iteration 2.0. 34 Selected classes from all 527 classes with 100% accuracy. mapping specified in ~./GoogleDrive/qualityEstimates_ForAllReratedClasses.xlsx###
#numOfLabels = 34 #Total number of labels selected from the original set of 537 classes
#mapping = np.zeros((numOfLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
#mapping[:, 0] = np.array([  0,   1,   2,   3,   5,   6,  16,  18,  23,  27,  47,  49,  66,
#        74,  75,  81, 111, 112, 113, 200, 288, 292, 300, 306, 307, 322,
#       327, 361, 371, 386, 396, 397, 418, 500])  #Specify all labels to be selected from original 527 classes
#mapping[:, 1] = np.arange(0, numOfLabels)

## Iteration 2.1. 30 Selected classes from all 527 classes with 100% accuracy. mapping specified in ~./GoogleDrive/Logistic-eg100k-iter100-batch512-classes34_100-confusionMatrix.xlsx###
numOfOrigLabels = 34 #Total number of labels selected from the original set of 537 classes
numOfLabels = 29    #Total number of new labels
mapping = np.zeros((numOfOrigLabels,2), dtype=int) # Mapping from the original 527 classes to the new set of labels
mapping[:, 0] = np.array([  0,   1,   2,   3,   5,   6,  16,  18,  23,  27,  47,  49,  66,
        74,  75,  81, 111, 112, 113, 200, 288, 292, 300, 306, 307, 322,
       327, 361, 371, 386, 396, 397, 418, 500])  #Specify all labels to be selected from original 527 classes
mapping[:, 1] = np.array([  0,   1,   2,   3,   4,   5,  6,   7,   8,   9,   10,  11,  12,
        13,  14,  15, 16,  16,  16,  17,  18,  19, 20,  20,  20,  21,
        22, 23, 24, 25, 26, 26,  27, 28])

def mapLabel(label):
    index = np.where(mapping[:,0] == label)[0][0]
    return mapping[index, 1]

def getOrigLabel(label):
    index = np.where(mapping[:,1] == label)[0][0]
    return mapping[index, 0]