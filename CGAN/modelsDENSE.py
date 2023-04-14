import os
from config import cla

PARAMS = cla()


os.environ['PYTHONHASHSEED']=str(PARAMS.seed_no)

import numpy as np
import matplotlib.pyplot as plt
import importlib
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from functools import partial


import random

random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
tf.random.set_seed(PARAMS.seed_no)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)

#reg_param   = 1e-5
# Act_func   = 'ReLU'
#Act_param  = 0.2
use_bias   = True


# if Act_func == 'ReLU':
#   activation_func = keras.layers.ReLU(negative_slope=Act_param)
# elif Act_func == 'tanh':  
#   activation_func = keras.layers.Activation('tanh')
# elif Act_func == 'sigmoid':  
#   activation_func = keras.layers.Activation('sigmoid') 
# elif Act_func == 'sin':
#   activation_func = tf.math.sin   



def CondInsNorm(input_X,input_Z,reg_param=1.0e-7,act_param=0.2):
  N,H,W,Nx = input_X.shape

  Nz = input_Z.shape[3]
  S1 = keras.layers.Conv2D(filters=Nx,
                           kernel_size=1,
                           strides=1,
                           padding='valid',
                           activation=keras.layers.ELU(alpha=1),
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias
                           )(input_Z)
  S2 = keras.layers.Conv2D(filters=Nx,
                           kernel_size=1,
                           strides=1,
                           padding='valid',
                           activation=keras.layers.ELU(alpha=1),
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias
                           )(input_Z)
  
  X  = keras.layers.Reshape((H*W,Nx), input_shape=(H,W,Nx))(input_X)
  Xs = keras.layers.Lambda(lambda x: tf.divide(x - tf.reduce_mean(x,axis=1,keepdims=True),
                                               1.0e-6+tf.sqrt(tf.math.reduce_variance(x,axis=1,keepdims=True))))(X)
  Xs = keras.layers.Reshape((H,W,Nx), input_shape=(H*W,Nx))(Xs)
  Xs = keras.layers.Multiply()([Xs,S1])
  Y  = keras.layers.Add()([Xs,S2])

  return Y

def DenseBlock(input_X,input_Z=None,normalization=None,reg_param=1.0e-7,act_param=0.2,out_channels=16,layers=4):
  N,H,W,Nx = input_X.shape

  padding = tf.constant([[0,0],[1,1],[1,1], [0, 0]])

  X = input_X
  for i in range(layers-1):
    if normalization =='cin':
      assert input_Z is not None
      X1 = CondInsNorm(input_X=X,input_Z=input_Z)
    elif normalization == 'in':  
   		X1 = tfa.layers.InstanceNormalization()(X)
    elif normalization == 'bn':
      X1 = keras.layers.BatchNormalization()(X)
    elif normalization == 'ln':
    	X1 = keras.layers.LayerNormalization()(X)
    else:
      X1=X 

    X1 = keras.layers.ELU(alpha=1)(X1)
    X1  = tf.pad(X1,padding,"REFLECT")

    X1  = keras.layers.Conv2D(filters=out_channels,
                             kernel_size=3,
                             strides=1,
                             padding='valid',
                             kernel_regularizer=keras.regularizers.l2(reg_param),
                             use_bias=use_bias
                             )(X1)
    #print(X.shape)
 #   (None, 32, 32, 4)
#     (None, 30, 30, 16)
    print(X1.shape) 
    X1=keras.layers.Concatenate(axis=-1)([X, X1])
    X=X1


  if normalization =='cin':
    assert input_Z is not None
    X1 = CondInsNorm(input_X=X,input_Z=input_Z)  
  elif normalization == 'in':  
    X1 = tfa.layers.InstanceNormalization()(X)  
  elif normalization == 'bn':
    X1 = keras.layers.BatchNormalization()(X)
  elif normalization == 'ln':
    X1 = keras.layers.LayerNormalization()(X)      
  else:
    X1=X

  X1 = keras.layers.ELU(alpha=1)(X1)
  X1  = tf.pad(X1,padding,"REFLECT")

  Y  = keras.layers.Conv2D(filters=Nx,
                             kernel_size=3,
                             strides=1,
                             padding='valid',
                             kernel_regularizer=keras.regularizers.l2(reg_param),
                             use_bias=use_bias
                             )(X1)     
  return Y
  

def DownSample(input_X,k,downsample=True,activation=True,reg_param=1.0e-7,act_param=0.2):

  padding = tf.constant([[0,0],[1,1],[1,1], [0, 0]])

  X  = tf.pad(input_X,padding,"REFLECT")
  X  = keras.layers.Conv2D(filters=k,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias)(X)
  print(X.shape)
  if activation:
    X = keras.layers.ELU(alpha=1)(X)                          
  if downsample:
    X = keras.layers.AveragePooling2D(pool_size=2,strides=2)(X)
                                                
  return X


def UpSample(input_X,k,old_X=None,concat=False,upsample=True,activation=True,reg_param=1.0e-7,act_param=0.2):

  padding = tf.constant([[0,0],[1,1],[1,1], [0, 0]])

  if concat:
    X = keras.layers.Concatenate()([input_X,old_X])
  else:
    X = input_X  

  X  = tf.pad(X,padding,"REFLECT")
  X  = keras.layers.Conv2D(filters=k,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias)(X)
  print(X.shape)
  if activation:
    X = keras.layers.ELU(alpha=1)(X)

  if upsample:
    X = keras.layers.UpSampling2D(size=2)(X)
                                                
  return X  

def generator(Input_W=64,Input_H=64,Input_C=1,Z_Dim=50,k0=32,reg_param=1.0e-7,act_param=0.2,out_channels=16,layers=3,ADDtanh=1,Etype=1):
  input_X = keras.Input(shape=(Input_W,Input_H,Input_C))
  input_Z = keras.Input(shape=(1,1,Z_Dim))

  # Downsampling + DenseBlock
  X1  = DownSample(input_X=input_X,k=k0,downsample=False,reg_param=reg_param,act_param=act_param)
  X1  = DenseBlock(input_X=X1,input_Z=input_Z,reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)
  
  X2  = DownSample(input_X=X1,k=2*k0,reg_param=reg_param,act_param=act_param)
  X2  = DenseBlock(input_X=X2,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  X3  = DownSample(input_X=X2,k=4*k0,reg_param=reg_param,act_param=act_param)
  X3  = DenseBlock(input_X=X3,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  # Final downsampling + ResBlock (not to be concatenated)
  X4  = DownSample(input_X=X3,k=8*k0,reg_param=reg_param,act_param=act_param)
  X4  = DenseBlock(input_X=X4,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  # Coarsest level DenseBlock
  X5  = DenseBlock(input_X=X4,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)
  
  # Upsampling + DenseBlock
  X6  = UpSample(input_X=X5,k=8*k0,reg_param=reg_param,act_param=act_param)
  X6  = DenseBlock(input_X=X6,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  # Upsampling + Concat + DenseBlock
  X7  = UpSample(input_X=X6,k=4*k0,concat=True,old_X=X3,reg_param=reg_param,act_param=act_param)
  X7  = DenseBlock(input_X=X7,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  X8  = UpSample(input_X=X7,k=2*k0,concat=True,old_X=X2,reg_param=reg_param,act_param=act_param)
  X8  = DenseBlock(input_X=X8,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  X9  = UpSample(input_X=X8,k=k0,concat=True,old_X=X1,upsample=False,reg_param=reg_param,act_param=act_param)
  X9  = DenseBlock(input_X=X9,input_Z=input_Z,normalization='cin',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  X10  = UpSample(input_X=X9,k=Input_C,upsample=False,activation=False,reg_param=reg_param,act_param=act_param)
  if ADDtanh==1:
    X11  = keras.activations.tanh(X10)
  else: 
    X11=X10
  
  X12  = X11+Etype*input_X
  model = keras.Model(inputs=[input_X, input_Z], outputs=X12)

  return model  

def critic(Input_W=64,Input_H=64,Input_C=1,k0=20,reg_param=1.0e-7,act_param=0.2,out_channels=16,layers=2):

  # We are asuming that X and Y have the same dimensions and same number of chanels
  input_XY = keras.Input(shape=(Input_W,Input_H,2*Input_C))

  # Downsampling + DenseBlock
  X1  = DownSample(input_X=input_XY,k=k0,downsample=False,reg_param=reg_param,act_param=act_param)
  X1  = DenseBlock(input_X=X1,reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)
  
  X2  = DownSample(input_X=X1,k=2*k0,reg_param=reg_param,act_param=act_param)
  X2  = DenseBlock(input_X=X2,normalization='ln',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  X3  = DownSample(input_X=X2,k=4*k0,reg_param=reg_param,act_param=act_param)
  X3  = DenseBlock(input_X=X3,normalization='ln',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  X4  = DownSample(input_X=X3,k=8*k0,reg_param=reg_param,act_param=act_param)
  X4  = DenseBlock(input_X=X4,normalization='ln',reg_param=reg_param,act_param=act_param,out_channels=out_channels,layers=layers)

  X5  = keras.layers.Flatten()(X4)

  X6  = keras.layers.Dense(units=128,
                           activation=keras.layers.ELU(alpha=1),
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias)(X5)
  X7  = keras.layers.LayerNormalization()(X6)                         
  X8  = keras.layers.Dense(units=1,
                           activation=None,
                           kernel_regularizer=keras.regularizers.l2(reg_param),
                           use_bias=use_bias)(X7)                         
  
  model = keras.Model(inputs=input_XY, outputs=X8)

  return model

    

def gradient_penalty(fake_X, true_X, true_Y,model, p=2):
    shape   = tf.concat((tf.shape(true_X)[0:1], tf.tile([1], [true_X.shape.ndims - 1])), axis=0)
    epsilon = tf.random.uniform(shape, 0.0, 1.0)
    #print(epsilon)
    x_hat   = epsilon * true_X + (1 - epsilon) * fake_X
    print(x_hat)
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat   = model(tf.concat([x_hat,true_Y],axis=3), training=True) 
    #print(d_hat)  
    gradients = t.gradient(d_hat, x_hat)
    ddx = tf.sqrt(1.0e-8 + tf.reduce_sum(tf.square(gradients), axis=tf.range(1,true_X.shape.ndims)))
    #print(ddx)
    d_regularizer = tf.reduce_mean(tf.pow(ddx-1.0,p))
    #print(d_regularizer)
    return d_regularizer 

def wt_bnds(model):
    max_wt = 0.0
    min_wt = 0.0
    for w in model.trainable_variables:
      max_wt = tf.math.maximum(max_wt,tf.reduce_max(w))
      min_wt = tf.math.minimum(min_wt,tf.reduce_min(w))

    return min_wt, max_wt   
