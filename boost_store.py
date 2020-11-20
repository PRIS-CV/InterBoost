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


def LM_load_model_img(mode = 0):
    X_train = np.load(open('LM_train_img.npy','rb'))
    count_train = 100
    Y_train = to_categorical(np.array([0] * count_train + [1] * count_train+[2] * count_train+\
        [3] * count_train+[4] *count_train+[5] * count_train+[6] *count_train+[7] * count_train))


    X_val = np.load(open('LM_val_img.npy','rb'))
    count_validation = 10
    Y_val = to_categorical(np.array([0] * count_validation + [1] *count_validation+\
        [2]*count_validation+[3]*count_validation+[4]*count_validation+[5]*count_validation+[6]*\
        count_validation+[7]*count_validation))        

    X_test = np.load(open('LM_test_img.npy','rb'))
    count_validation = 100
    Y_test = to_categorical(np.array([0] * count_validation + [1] *count_validation+\
        [2]*count_validation+[3]*count_validation+[4]*count_validation+[5]*count_validation+[6]*\
        count_validation+[7]*count_validation))
    
    index = [i for i in range(len(X_test))]  
    random.shuffle(index) 
    X_test = X_test[index]
    Y_test = Y_test[index]   

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    

    model.add(Flatten())
    model.add(Dense(8, name='output_1'))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))


    return  model, X_train, Y_train, X_val, Y_val, X_test, Y_test  
def UIUC_load_model_img(mode = 0):
    X_train = np.load(open('UIUC_train_img.npy','rb'))
    count_train = [95,63,113,86,92,120,90,90]
    Y_train = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    

    X_val = np.load(open('UIUC_val_img.npy','rb'))
    count_train = [10] * 8
    Y_val = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    
    X_test = np.load(open('UIUC_test_img.npy','rb'))
    count_train = [95,63,113,86,92,120,90,90]
    Y_test = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    
    index = [i for i in range(len(X_test))]  
    random.shuffle(index) 
    X_test = X_test[index]
    Y_test = Y_test[index]   

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    

    model.add(Flatten())

    model.add(Dense(8, name='output_1'))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))


    return  model, X_train, Y_train, X_val, Y_val, X_test, Y_test  
def LM_load_model_VGG(mode = 0):
    X_train = np.load(open('LM_features_train.npy','rb'))
    count_train = 100
    Y_train = to_categorical(np.array([0] * count_train + [1] * count_train+[2] * count_train+\
        [3] * count_train+[4] *count_train+[5] * count_train+[6] *count_train+[7] * count_train))


    X_val = np.load(open('LM_features_validation.npy','rb'))
    count_validation = 10
    Y_val = to_categorical(np.array([0] * count_validation + [1] *count_validation+\
        [2]*count_validation+[3]*count_validation+[4]*count_validation+[5]*count_validation+[6]*\
        count_validation+[7]*count_validation))        

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
    model_1.add(Dense(32, activation='relu', name='relu', W_regularizer=l2(0.01)))
    model_1.add(Dense(8, name='output_1', W_regularizer=l2(0.01)))
    model_1.add(Activation("softmax"))

    return  model_1, X_train, Y_train, X_val, Y_val, X_test, Y_test  



def UIUC_load_model_VGG(mode = 0):
    X_train = np.load(open('UIUC_features_train.npy','rb'))
    count_train =[95,63,113,86,92,120,90,90]
    Y_train = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    

    X_val = np.load(open('UIUC_features_validation.npy','rb'))
    count_train = [10] * 8
    Y_val = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    



    X_test = np.load(open('UIUC_features_test.npy','rb'))
    count_train =[95,63,113,86,92,120,90,90]
    Y_test = to_categorical(np.array([0] * count_train[0] + [1] * count_train[1]+[2] * count_train[2]+ \
        [3] * count_train[3]+[4] *count_train[4]+[5] * count_train[5]+[6] *count_train[6]+[7] * count_train[7]))
    
    index = [i for i in range(len(X_test))]  
    random.shuffle(index) 
    X_test = X_test[index]
    Y_test = Y_test[index]

    model_1 = Sequential()
    model_1.add(Flatten(input_shape=X_train.shape[1:]))
    model_1.add(Dense(32, activation='relu', name='relu', W_regularizer=l2(0.01)))
    model_1.add(Dense(8, name='output_1', W_regularizer=l2(0.01)))


    model_1.add(Activation("softmax"))

    return  model_1, X_train, Y_train, X_val, Y_val, X_test, Y_test   




