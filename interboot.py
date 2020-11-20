#encoding=utf-8
import os
import sys
import random
import shutil
import argparse
import numpy as np
import tensorflow as tf
import random as rn
from boost_store import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint 
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import SGD, rmsprop,Adam
import keras.callbacks as callbacks
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():  
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement=True
    return tf.Session(config=sess_config)
KTF.set_session(get_session())  

def get_ensemble_acc(model_1_pro,model_2_pro,p_true=0, num = 0):
    pro = model_1_pro + model_2_pro
    pro_output = np.argmax(pro,axis = -1)
    pro_count = 0
    P_true = np.argmax(p_true,axis = -1)

    for i in range(num):
        if pro_output[i] == P_true[i]:
            pro_count+=1

    return pro_count / (num * 1.0)

def run(batch_size,
        epochs,
        weights,
        loops,
        lr):
    """
        batch_size: Number of samples per gradient update.
        epochs:  total number of epochs that the model will be trained for.
        weights: model weight Floder.
        loops:   total number of loops that the model will be trained for.
        lr:      float >= 0. Learning rate.
    """
    if not os.path.exists(weights):
        os.makedirs(weights)


    select_val_ensemble_acc = 0
    select_ensemble_val_loss = 10000000
    cus_rms =rmsprop(lr=lr)


    model_1,x_train,y_train,x_val,y_val,x_test, y_test = LM_load_model_VGG()
    model_1.compile(loss='categorical_crossentropy',
                      optimizer=cus_rms,
                      metrics=['accuracy'])


    model_2,x_train,y_train,x_val,y_val,x_test, y_test = LM_load_model_VGG()
    model_2.compile(loss='categorical_crossentropy',
              optimizer=cus_rms,
              metrics=['accuracy'])

################################################################################################
    x_val = np.array(x_train.tolist()+x_val.tolist())
    y_val = np.array(y_train.tolist()+y_val.tolist())

###############################################################################################
    sample_1 =  np.array([random.random() for i in range(len(y_train))])
    sample_2 =  np.array([1-i for i in sample_1])

    # sample_1 =  np.array([1. for i in range(len(y_train))])
    # sample_2 =  np.array([1. for i in range(len(y_train))])

    for count in range(1,loops + 1,1):

        if count > 1:
            try:
                model_1.load_weights(weights+'model_1_temp.h5')
                model_2.load_weights(weights+'model_2_temp.h5')
                print("load_model_weights")
            except:
                raise "load_weights fail"
        

        checkpointer = ModelCheckpoint(filepath=weights+"model_1.h5", monitor='val_acc',save_best_only=True, mode='max')
        
        checkpointer2 = ModelCheckpoint(filepath=weights+"model_2.h5", monitor='val_acc',save_best_only=True, mode='max')
        

        loop_callbacks_1  = [checkpointer]
        loop_callbacks_2  = [checkpointer2]            

        model_1.fit(x_train, y_train,
                batch_size=batch_size,epochs=epochs,validation_data=(x_val, y_val),verbose = 2,
            callbacks=loop_callbacks_1,
            sample_weight=sample_1)
                 
        model_2.fit(x_train, y_train,
                     batch_size=batch_size,epochs=epochs,validation_data=(x_val, y_val),verbose = 2,
                  callbacks=loop_callbacks_2,
                  sample_weight=sample_2)                
                  
        model_1.load_weights(weights+'model_1.h5')
        model_2.load_weights(weights+'model_2.h5')                
                  
        model_1_prob = model_1.predict(x_train, batch_size=batch_size)
        model_2_prob = model_2.predict(x_train, batch_size=batch_size)

        model_1_prob = np.sum(model_1_prob * y_train, axis=-1)
        model_2_prob = np.sum(model_2_prob * y_train, axis=-1)


        model_1_sample = range(0,len(y_train))
        model_2_sample = range(0,len(y_train))


        for i in range(len(y_train)):
            model_1_sample[i] = np.log(model_1_prob[i]) / (np.log(model_1_prob[i]) + np.log(model_2_prob[i] + 1e-10))
            model_2_sample[i] = np.log(model_2_prob[i]) / (np.log(model_1_prob[i]) + np.log(model_2_prob[i] + 1e-10))  

        sample_1 = np.array(model_1_sample)
        sample_2 = np.array(model_2_sample)


        model_1_pro = model_1.predict(x_val, batch_size=batch_size) 
        model_2_pro = model_2.predict(x_val, batch_size=batch_size) 

        emsemble_val_acc = get_ensemble_acc(model_1_pro,model_2_pro,p_true=y_val,num = len(x_val))
        

        shutil.copyfile(weights+"model_1.h5", weights+"model_1_temp.h5")
        shutil.copyfile(weights+"model_2.h5", weights+"model_2_temp.h5")    

        a_val_loss,a_val_acc = model_1.evaluate(x_val,y_val)
        b_val_loss,b_val_acc = model_2.evaluate(x_val,y_val)

        ensemble_val_loss = a_val_loss + b_val_loss

              
        if  ensemble_val_loss <=  select_ensemble_val_loss :
            select_ensemble_val_loss = ensemble_val_loss
            select_val_ensemble_acc  = emsemble_val_acc

            os.rename(weights+"model_1.h5", weights+'model_1_the_best_weights.h5')
            os.rename(weights+"model_2.h5", weights+'model_2_the_best_weights.h5')
        else:
            os.remove(weights+'model_1.h5')
            os.remove(weights+'model_2.h5')               


        model_1_pro = model_1.predict(x_test, batch_size=batch_size) 
        model_2_pro = model_2.predict(x_test, batch_size=batch_size)

        ensemble_test_acc = get_ensemble_acc(model_1_pro,model_2_pro,p_true=y_test,num = len(y_test))

        detail = {"ensemble_val_loss":ensemble_val_loss,"count":count,"ensemble_test_acc":ensemble_test_acc,
        "a_val_acc":a_val_acc,"b_val_acc":b_val_acc,"ensemble_val_acc":emsemble_val_acc}

        f = file(weights+"ensemble_test_acc.txt","a")
        f.write(str(detail)+",")
        f.close()

    model_1.load_weights(weights+'model_1_the_best_weights.h5')
    model_2.load_weights(weights+'model_2_the_best_weights.h5')



    model_1_pro = model_1.predict(x_test, batch_size=batch_size) 
    model_2_pro = model_2.predict(x_test, batch_size=batch_size)


    ensemble_test_acc = get_ensemble_acc(model_1_pro,model_2_pro,p_true=y_test,num = len(y_test))


    a_val_acc = model_1.evaluate(x_val,y_val)[1]
    b_val_acc = model_2.evaluate(x_val,y_val)[1]
    emsemble_val_acc = select_val_ensemble_acc

    a_acc = model_1.evaluate(x_test,y_test)[1]
    b_acc = model_2.evaluate(x_test,y_test)[1]


    detail = {"0Final":"Final","a_acc":a_acc,"b_acc":b_acc,"ensemble_test_acc":ensemble_test_acc,
    "a_val_acc":a_val_acc,"b_val_acc":b_val_acc,"ensemble_val_acc":emsemble_val_acc}

    f = file(weights+"ensemble_test_acc.txt","a")
    f.write(str(detail)+",")

    f.close()


    f = file(weights+"Final_test_acc.txt","a")
    f.write(str(ensemble_test_acc)+",")

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
    parser.add_argument('--weights',  type=str,default="Ours/",
                    help='model weight Floder. ')
    parser.add_argument('--lr',  type=int,default=0.001,
                help='Learning rate')
    args = parser.parse_args()

    print(args)

    for i in range(args.round_num):
        run(batch_size = args.batch_size,
            epochs     = args.epochs,
            weights    = args.weights,
            loops      = args.loops,
            lr         = args.lr)  
              

          
              
              
      
      