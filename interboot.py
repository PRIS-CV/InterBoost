#encoding=utf-8
import os
import sys
import shutil
import random
import argparse
import numpy as np
import random as rn
from scipy import *  
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import keras.callbacks as callbacks
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from keras.layers import Dense, Input, Flatten
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import SGD, rmsprop,Adam
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



from boost_store import *

# def get_session():  
#     sess_config = tf.ConfigProto()
#     sess_config.gpu_options.allow_growth = True
#     sess_config.allow_soft_placement=True
#     return tf.Session(config=sess_config)
# KTF.set_session(get_session())  



gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_ensemble_acc(model_1_pro,model_2_pro,p_true=0, num = 0):
    pro = model_1_pro + model_2_pro
    pro_output = np.argmax(pro,axis = -1)
    pro_count = 0
    P_true = np.argmax(p_true,axis = -1)

    for i in list(range(num)):
        if pro_output[i] == P_true[i]:
            pro_count+=1

    return pro_count / (num * 1.0)

def run(batch_size,
        epochs,
        weights,
        loops,
        lr,
        l2_val):
    """
        batch_size: Number of samples per gradient update.
        epochs:  total number of epochs that the model will be trained for.
        weights: model weight Floder.
        loops:   total number of loops that the model will be trained for.
        lr:      float >= 0. Learning rate.
    """




    select_ensemble_loss = 10000000
    pass_loop = False

    cus_rms  =rmsprop(lr=lr)
    cus_rms2 =rmsprop(lr=lr)


    model_1,x_train,y_train,x_test, y_test = LM_load_model_VGG()
    model_1.compile(loss='categorical_crossentropy',
                      optimizer=cus_rms,
                      metrics=['accuracy'])


    model_2,x_train,y_train,x_test, y_test = LM_load_model_VGG()
    model_2.compile(loss='categorical_crossentropy',
              optimizer=cus_rms2,
              metrics=['accuracy'])



    sample_1 =  np.array([random.random() for i in list(range(len(y_train)))])
    sample_2 =  np.array([1-i for i in sample_1])

    detail_info = []
    for count in list(range(0,loops)):
        print("-----loop-----",count)
        if count > 0:
            try:
                model_1.load_weights(weights+'model_1_temp.h5')
                model_2.load_weights(weights+'model_2_temp.h5')
                print("load_model_weights")
            except:
                raise "load_weights fail"
        

        checkpointer  = ModelCheckpoint(filepath=weights+"model_1.h5", monitor='loss',save_best_only=True, mode='min')
        
        checkpointer2 = ModelCheckpoint(filepath=weights+"model_2.h5", monitor='loss',save_best_only=True, mode='min')
        



####
        def cosine_anneal_schedule(t):
            cos_inner = np.pi * (t % (epochs ))  # t - 1 is used when t has 1-based indexing.
            cos_inner /= epochs
            cos_out = np.cos(cos_inner) + 1
            return float(lr / 2 * cos_out)
####

        loop_callbacks_1  = [checkpointer,
                            callbacks.LearningRateScheduler(schedule=cosine_anneal_schedule)]
        loop_callbacks_2  = [checkpointer2,
                            callbacks.LearningRateScheduler(schedule=cosine_anneal_schedule)]            

        hist_1 = model_1.fit(x_train, y_train,
                            batch_size=batch_size,epochs=epochs,verbose = 2,
                        callbacks=loop_callbacks_1,
                        sample_weight=sample_1)
                     
        hist_2 = model_2.fit(x_train, y_train,
                         batch_size=batch_size,epochs=epochs,verbose = 2,
                      callbacks=loop_callbacks_2,
                      sample_weight=sample_2)           

        # import pdb
        # pdb.set_trace()

        acc_1 = np.max(hist_1.history["accuracy"])  
        acc_2 = np.max(hist_2.history["accuracy"]) 
        # if acc_1<1. or acc_2<1.:
        #     pass_loop = True

        #     print("restart loops")
        #     break

        model_1.load_weights(weights+'model_1.h5')
        model_2.load_weights(weights+'model_2.h5')                
                  
        model_1_prob = model_1.predict(x_train, batch_size=batch_size)
        model_2_prob = model_2.predict(x_train, batch_size=batch_size)

        model_prob = (model_1_prob + model_2_prob)/2.
        model_loss = - np.mean(y_train* np.log(model_prob))

        model_1_prob = np.sum(model_1_prob * y_train, axis=-1)
        model_2_prob = np.sum(model_2_prob * y_train, axis=-1)



#################################should del
        amodel_1_pro = model_1.predict(x_test, batch_size=batch_size) 
        bmodel_2_pro = model_2.predict(x_test, batch_size=batch_size)


        ensemble_test_acc = get_ensemble_acc(amodel_1_pro,bmodel_2_pro,p_true=y_test,num = len(y_test))

        detail_info.append({"net_loss":model_loss,"test_acc":ensemble_test_acc})

##################################
        model_1_sample = list(range(0,len(y_train)))
        model_2_sample = list(range(0,len(y_train)))

        model_1_prob+=1e-10
        model_2_prob+=1e-10
        for i in list(range(len(y_train))):
            model_1_sample[i] = np.log(model_1_prob[i]) / (np.log(model_1_prob[i]) + np.log(model_2_prob[i] )+ 1e-10)
            model_2_sample[i] = np.log(model_2_prob[i]) / (np.log(model_1_prob[i]) + np.log(model_2_prob[i] )+ 1e-10)  



        sample_1 = np.array(model_1_sample)
        sample_2 = np.array(model_2_sample)

        shutil.copyfile(weights+"model_1.h5", weights+ "model_1_temp.h5")
        shutil.copyfile(weights+"model_2.h5", weights+ "model_2_temp.h5")  

              
        if  model_loss <=  select_ensemble_loss :
            select_ensemble_loss = model_loss
            os.rename(weights+"model_1.h5", weights+'model_1_the_best_weights.h5')
            os.rename(weights+"model_2.h5", weights+'model_2_the_best_weights.h5')
        else:
            os.remove(weights+'model_1.h5')
            os.remove(weights+'model_2.h5')               


    print("-----------------------------------")
    print("testing")
    if pass_loop == False:
        model_1.load_weights(weights+'model_1_the_best_weights.h5')
        model_2.load_weights(weights+'model_2_the_best_weights.h5')



        model_1_pro = model_1.predict(x_test, batch_size=batch_size) 
        model_2_pro = model_2.predict(x_test, batch_size=batch_size)


        ensemble_test_acc = get_ensemble_acc(model_1_pro,model_2_pro,p_true=y_test,num = len(y_test))


        f = open(weights+"Final_test_acc.txt","a")
        f.write(str(ensemble_test_acc)+",")

        f.close()

        f = open(weights+"detail_info.txt","a")
        f.write(str(detail_info)+",")

        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--batch_size',  type=int,default=32,
                        help='Batch size')
    parser.add_argument('--epochs',  type=int,default=20,
                    help='epochs')
    parser.add_argument('--loops',  type=int,default=5,
                    help='total number of loops')
    parser.add_argument('--round_num',  type=int,default=60,
                    help='total number of round_num')
    parser.add_argument('--weights',  type=str,default="interboost/",
                    help='model weight Floder. ')
    parser.add_argument('--lr',  type=int,default=0.001,
                help='Learning rate')
    parser.add_argument('--l2_val',  type=int,default=0.001,
                help='l2_val')
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.weights):
        os.makedirs(args.weights)
        
    np.save(args.weights+"all_loops",0)
    for i in list(range(args.round_num)):

        run(batch_size = args.batch_size,
            epochs     = args.epochs,
            weights    = args.weights,
            loops      = args.loops,
            lr         = args.lr,
            l2_val     = args.l2_val)  
              
              