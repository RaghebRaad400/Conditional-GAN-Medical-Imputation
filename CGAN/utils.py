import os
from config import cla

PARAMS = cla()


os.environ['PYTHONHASHSEED']=str(PARAMS.seed_no)

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import h5py
import scipy.io
from os.path import dirname, join as pjoin
import mat73
from config import cla
from modelsDENSE import *


import random


random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
tf.random.set_seed(PARAMS.seed_no)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)


#============== Parameters ======================
def read_images_mat(img_dir,N_train,batch_size,drop_remainder=True):

    
    # Loading training data
    filename = img_dir + '/training_data-1to1REG.mat'
    #filename = img_dir + '/training_data.mat'

    assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
    #file = h5py.File(filename, "r+")
    #train_images = file["/XYtraining"][0::]
    #file.close()
    
    
    mat_fname = pjoin(img_dir, 'training_data-1to1REG.mat')
    #mat_fname = pjoin(img_dir, 'training_data.mat')

    train_imagesdict=mat73.loadmat(mat_fname)
    #mat_fname = pjoin(img_dir, 'SmallSet.mat')
    #train_imagesdict=scipy.io.loadmat(mat_fname)
    #train_images=train_imagesdict['SmallTrainingSet']
    train_images=train_imagesdict['XYtraining']    
    #train_images=train_images[0:1000,:,:,:]    #comment later
    print(train_images.shape)
    
    
    # Loading validatiton data
    filename = img_dir + '/validation_data-1to1REG.mat'
   # filename = img_dir + '/validation_dataNEW.mat'

    assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
    #file = h5py.File(filename, "r+")
    #valid_images = file["/XYvalidation"][0::]
    #file.close()
    
    mat_fname = pjoin(img_dir, 'validation_data-1to1REG.mat')
   # mat_fname = pjoin(img_dir, 'validation_dataNEW.mat')

    valid_imagesdict=scipy.io.loadmat(mat_fname)
    valid_images=valid_imagesdict['XYvalidation']
   # valid_images=valid_images[0:100,:,:,:]    #comment later

    total_train  = len(train_images)
    print(train_images.shape)
    assert total_train >= N_train
    train_images = train_images[0:N_train]
    
    # We are assuming that the data is saved with chanel >= 2 (1 chanel at least each for X and Y)
    width,height,chanels = train_images[0].shape
    chanels = int(chanels/2)
    
    train_images=tf.cast(train_images,tf.float32)
    valid_images=tf.cast(valid_images,tf.float32)
    #print(train_images)
    N_valid      = len(valid_images)

    print(f'     *** Datasets:')
    print(f'         ... training samples   = {N_train} of {total_train}')
    print(f'         ... validation samples = {N_valid}')
    print(f'         ... sample dimension   = {width}X{height}X{chanels}')

    train_data = tf.data.Dataset.from_tensor_slices(train_images).shuffle(N_train).batch(batch_size, drop_remainder=True)
    #print(train_data)
    #train_data = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size, drop_remainder=True)

    return train_data, valid_images, N_valid, width, height,chanels
    
    

def read_images_hdf5(img_dir,N_train,batch_size,drop_remainder=True):

    
    # Loading training data
    filename = img_dir + '/training_data.h5'
    assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
    file = h5py.File(filename, "r+")
    train_images = file["/images"][0::]
    file.close()

    # Loading validatiton data
    filename = img_dir + '/validation_data.h5'
    assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
    file = h5py.File(filename, "r+")
    valid_images = file["/images"][0::]
    file.close()


    total_train  = len(train_images)
    assert total_train >= N_train
    train_images = train_images[0:N_train]
    
    # We are assuming that the data is saved with chanel >= 2 (1 chanel at least each for X and Y)
    width,height,chanels = train_images[0].shape
    chanels = int(chanels/2)

    N_valid      = len(valid_images)

    print(f'     *** Datasets:')
    print(f'         ... training samples   = {N_train} of {total_train}')
    print(f'         ... validation samples = {N_valid}')
    print(f'         ... sample dimension   = {width}X{height}X{chanels}')

   # train_data = tf.data.Dataset.from_tensor_slices(train_images).shuffle(N_train).batch(batch_size, drop_remainder=True)
    train_data = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size, drop_remainder=True)


    return train_data, valid_images, N_valid, width, height,chanels
def read_test_images_mat(img_dir):

    
    # Loading validation data
    #filename = img_dir + '/validation_data.h5'
    #assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
    #file = h5py.File(filename, "r+")
    #test_images = file["/images"][0::]
    
    
       # Loading validatiton data
    filename = img_dir + '/validation_data-1to1REG.mat'
   # filename = img_dir + '/CutTraining.mat'
    assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
    #file = h5py.File(filename, "r+")
    #valid_images = file["/XYvalidation"][0::]
    #file.close()
    
    mat_fname = pjoin(img_dir, 'validation_data-1to1REG.mat')
    valid_imagesdict=scipy.io.loadmat(mat_fname)
    test_images=valid_imagesdict['XYvalidation']
   # test_images=valid_imagesdict['CutTraining']

    
    N = len(test_images)
    width,height,chanels= test_images[0].shape
    chanels = int(chanels/2)
    print(f'Datasets:')
    print(f'... testing/validation samples    = {N}')

    return test_images, N, width, height ,chanels
    
    
def read_test_images_hdf5(img_dir):

    
    # Loading training data
    filename = img_dir + '/validation_data.h5'
    assert os.path.exists(filename), 'Error: The data file ' + filename + ' is unavailable.'
    file = h5py.File(filename, "r+")
    test_images = file["/images"][0::]
    file.close()

    N = len(test_images)
    width,height,_ = test_images[0].shape

    print(f'Datasets:')
    print(f'... testing/validation samples    = {N}')

    return test_images, N, width, height    

def get_lat_var(batch_size,x_W,x_H,z_dim):
    z = tf.random.normal((batch_size,1,1,z_dim))
    return z    


def save_loss(loss,loss_name,savedir,n_epoch):
    

    np.savetxt(f"{savedir}/{loss_name}.txt",loss)
    fig, ax1 = plt.subplots()
    ax1.plot(loss)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1,len(loss)])


    ax2 = ax1.twiny()
    ax2.set_xlim([0,n_epoch])
    ax2.set_xlabel('Epochs')


    plt.tight_layout()
    plt.savefig(f"{savedir}/{loss_name}.png",dpi=200)    
    plt.close()  

def get_main_dir(dir_path):

    n = len(dir_path)-1
    found = False
    while n > - 1 and not found:
        if dir_path[n] != '/':
            n-=1
        else:
            found = True
            n+=1

    return dir_path[n::]       


    
