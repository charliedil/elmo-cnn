import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import pickle
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.engine import Layer
import numpy as np
from elmo_embedding_layer import ElmoEmbeddingLayer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
logFile = open("logFile","w")
trainFile = open("../data/sentence_train","r")
labelFile = open("../data/labels_train","r")
trainData = trainFile.read().split("\n")
labelData = labelFile.read().split("\n")
for i in range(len(labelData)):
    if(labelData[i]!=""):
        labelData[i]=[labelData[i]]
    else:
        labelData.remove(labelData[i])

for i in range(len(trainData)):
    if trainData[i]=='':
        trainData.remove(trainData[i])

trainFile.close()
labelFile.close()
logFile.write("splitting and shuffling data, 80-20\n")
X_train, X_test,y_train,y_test =train_test_split(trainData,labelData,test_size=.2,random_state=42)
logFile.write("length of X_train is: "+str(len(X_train))+"\n")
logFile.write("length of X_test is: "+str(len(X_test))+"\n")
logFile.write("total amount of sentences is " +str(len(X_test)+len(X_train))+"\n")
mlb=MultiLabelBinarizer()
mlb.fit(labelData)
y_test=mlb.transform(y_test)
y_train=mlb.transform(y_train)

X_test = np.array(X_test,dtype=object)[:,np.newaxis]
X_train = np.array(X_train,dtype=object)[:,np.newaxis]
def build_model():

    input_text = Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    conv = Conv1D(filters=250, kernel_size=3, activation='relu',padding='same')(embedding)
    pool = MaxPooling1D()(conv)
    ###conv1 = Conv1D(filters=32, kernel_size=1, activation='relu')(pool)
    ###drop = Dropout(.5)(conv1)
    flat = Flatten()(conv)
    dense = Dense(32,activation='relu')(flat)
    pred = Dense(8, activation='softmax')(dense)

    model=Model(inputs=[input_text], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model
#80-20 split?
logFile.write("Begin training model...\n")
model = build_model()
model.fit(X_train, y_train,epochs=3,batch_size=1)
logFile.write("Model training finished, now writing model weights to file\n")

model.save("weights/n2c2/elmo-cnn-single-v1-3e.h5") 
logFile.write("Write successful, now testing load\n")
logFile.write("Loading weights into model\n")
thing = build_model()
thing.load_weights("weights/n2c2/elmo-cnn-single-v1-3e.h5")
logFile.write("Load successful, now predicting on X_test\n")
binarizedPredictions=thing.predict(X_test,batch_size=1)
logFile.write("predictions successful, now writing classification report\n")

for i in range(len(mlb.classes_)):
    target_names = ["Other",mlb.classes_[i]]
    labels = [0,1]
    pred = []
    actual = []
    logFile.write("\n")
    logFile.write(str(mlb.classes_[i])+ " REPORT---------------------------------------\n")
    for p in binarizedPredictions:
        if(np.argmax(p,axis=0)==i):#OR WHATEVER YOU ARE USING TO DETERMINE LABELS!!!
            pred.append(1)
        else:
            pred.append(0)
    for a in y_test:
        if(a[i]==1):
            actual.append(1)
        else:
            actual.append(0)
    logFile.write(classification_report(actual,pred,target_names=target_names,labels=labels))
#print(classification_report(y_test[0],binarizedPredictions[0],target_names=mlb.classes_,labels=[0,1]))
logFile.write("\n")
logFile.write("SUCCESS- now exiting\n")
logFile.close()
