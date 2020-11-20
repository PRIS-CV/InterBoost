import numpy as np
import random
import os
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
img_width, img_height = 256,256


def LM_load_model_VGG(l2_val=0.0001,sum_train=0):
    if sum_train == 0:
        X_train = np.load(open('LM_train_0.npy','rb'))
        count_train = 100
    if sum_train == 1:
        X_train = np.load(open('LM_train_1.npy','rb'))
        count_train = 90
    if sum_train == 2:
        X_train = np.load(open('LM_train_2.npy','rb'))
        count_train = 70
    if sum_train == 3:
        X_train = np.load(open('LM_train_3.npy','rb'))
        count_train = 50
    if sum_train == 4:
        X_train = np.load(open('LM_train_4.npy','rb'))
        count_train = 30

    Y_train = to_categorical(np.array([0] * count_train + [1] * count_train+[2] * count_train+\
        [3] * count_train+[4] *count_train+[5] * count_train+[6] *count_train+[7] * count_train))


    X_test = np.load(open('LM_features_test.npy','rb'))
    count_validation = 100
    Y_test = to_categorical(np.array([0] * count_validation + [1] *count_validation+\
        [2]*count_validation+[3]*count_validation+[4]*count_validation+[5]*count_validation+[6]*\
        count_validation+[7]*count_validation))
    
    index = [i for i in range(len(X_test))]  
    random.shuffle(index) 
    X_test = X_test[index]
    Y_test = Y_test[index]   


    model_1 = Sequential()
    model_1.add(Flatten(input_shape=X_train.shape[1:]))
    model_1.add(Dense(32, activation='relu', name='relu', W_regularizer=l2(l2_val)))
    model_1.add(Dense(8, name='output_1', W_regularizer=l2(l2_val)))
    model_1.add(Activation("softmax"))

    return  model_1, X_train, Y_train, X_test, Y_test  



def UIUC_load_model_VGG(l2_val=0.0001,sum_train=0):

    if sum_train == 0:
        X_train = np.load(open('UIUC_train_0.npy','rb'))
        count_train = [92,95,63,113,86,120,90,90]
    if sum_train == 1:
        X_train = np.load(open('UIUC_train_1.npy','rb'))
        count_train = [72, 75, 43, 93, 66, 100, 70, 70]
    if sum_train == 2:
        X_train = np.load(open('UIUC_train_2.npy','rb'))
        count_train = [62, 65, 33, 83, 56, 90, 60, 60]
    if sum_train == 3:
        X_train = np.load(open('UIUC_train_3.npy','rb'))
        count_train = [52, 55, 23, 73, 46, 80, 50, 50]
    if sum_train == 4:
        X_train = np.load(open('UIUC_train_4.npy','rb'))
        count_train = [42, 45, 13, 63, 36, 70, 40, 40]



    Y_train = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    

    X_val = np.load(open('UIUC_features_validation.npy','rb'))
    count_train = [5] * 8
    Y_val = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))

    foo = X_val
    foo_1 = []
    foo_2 = []
    for i in range(8):
        foo_1.append(foo[i*10:(i+1)*10][:5])
        foo_2.append(foo[i*10:(i+1)*10][5:])
    foo_1 = np.array(foo_1).reshape(40,512,8,8)
    foo_2 = np.array(foo_2).reshape(40,512,8,8)






    X_test = np.load(open('UIUC_features_test.npy','rb'))
    count_train =[92,95,63,113,86,120,90,90]
    Y_test = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    

    #####
    X_train = np.array(X_train.tolist()+foo_1.tolist())
    Y_train = np.array(Y_train.tolist()+Y_val.tolist())
    #####
    X_test = np.array(X_test.tolist()+foo_2.tolist())
    Y_test = np.array(Y_test.tolist()+Y_val.tolist())



    index = [i for i in range(len(X_test))]  
    random.shuffle(index) 
    X_test = X_test[index]
    Y_test = Y_test[index]

    model_1 = Sequential()
    model_1.add(Flatten(input_shape=X_train.shape[1:]))
    model_1.add(Dense(32, activation='relu', name='relu', W_regularizer=l2(l2_val)))
    model_1.add(Dense(8, name='output_1', W_regularizer=l2(l2_val)))


    model_1.add(Activation("softmax"))

    return  model_1, X_train, Y_train,X_test, Y_test   




def scene_15_load_model_VGG(l2_val=0.0001,sum_train=0):

    X_train = np.load(open('15_scene_features_train.npy','rb'))


    train_labels = []
    num_classes= [120,180,164,130,154,187,205,146,178,107,108,155,105,144,157]
    for i in range(15):
        train_labels += [i] * num_classes[i]
    train_labels = np.array(train_labels)
    Y_train = to_categorical(train_labels)



    X_test = np.load(open('15_scene_features_test.npy','rb'))
    test_labels = []
    num_classes= [120,180,164,130,154,187,205,146,178,107,108,155,105,144,157]
    for i in range(15):
        test_labels += [i] * num_classes[i]
    test_labels = np.array(test_labels)
    Y_test = to_categorical(test_labels)
 


    model_1 = Sequential()
    model_1.add(Flatten(input_shape=X_train.shape[1:]))
    model_1.add(Dense(64, activation='relu', name='relu', W_regularizer=l2(l2_val)))
    model_1.add(Dense(15, name='output_1', W_regularizer=l2(l2_val)))
    model_1.add(Activation("softmax"))

    return  model_1, X_train, Y_train, X_test, Y_test  


def caltech_101_load_model_VGG(l2_val=0.0001,sum_train=0):

    X_train = np.load(open('caltech101-features-train.npy','rb'))


    train_labels = []
    num_classes= np.load("caltech101-target.npy")
    for i in range(101):
        train_labels += [i] * num_classes[i]
    train_labels = np.array(train_labels)
    Y_train = to_categorical(train_labels)



    X_test = np.load(open('caltech101-features-test.npy','rb'))
    test_labels = []
    num_classes= np.load("caltech101-target.npy")
    for i in range(101):
        test_labels += [i] * num_classes[i]
    test_labels = np.array(test_labels)
    Y_test = to_categorical(test_labels)
 


    model_1 = Sequential()
    model_1.add(Flatten(input_shape=X_train.shape[1:]))
    model_1.add(Dense(512, activation='relu', name='relu', W_regularizer=l2(l2_val)))
    model_1.add(Dense(101, name='output_1', W_regularizer=l2(l2_val)))
    model_1.add(Activation("softmax"))

    return  model_1, X_train, Y_train, X_test, Y_test  