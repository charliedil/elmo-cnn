# elmo-cnn
CNN with trainable ELMo layer for Relationship Extraction
## Usage
To use this package, you need to have your sentences in a file separated by new lines. This is the data file which will be split into 80-20 train-test split. The labels for these sentences need to be in a separate labels file. The paths to these files can be inputed at lines 17 to 18
```
trainFile = open("../data/sentence_train","r")
labelFile = open("../data/labels_train","r")
```
The output of this program is a model file for which the path can be changed at lines 67 and 71. Here is line 67:
```
model.save("weights/n2c2/elmo-cnn-single-v1-3e.h5") 
```
Also an output of this program is a log file which contains Precision Recall and F1 scores for the model's performance on the test set. The path can be changed at line 16
```
logFile = open("logFile","w")
```
