import math
import keras
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Nadam, SGD
from keras.utils import multi_gpu_model

import tensorflow as tf

import utils.regularisers
import utils.losses
import utils.layers
import utils.optimisers

def applyConvolutions(events, convolutions):
  
  outputs = []
  for c in convolutions:
    filtered = c(events)
    outputs.append(filtered)
  
  maxOutput = keras.layers.maximum(outputs)
  maskInput = keras.layers.concatenate(outputs, axis=3)
  maskOutput = utils.layers.FilterMask()(maskInput)

  return maxOutput, maskOutput

def applyDeconvolutions(events, deconvolutions, mask):
  
  applied = []
  
  for d in deconvolutions:
    applied.append(d(events))
  
  events = keras.layers.concatenate(applied, axis=3)
  events = keras.layers.multiply([events, mask])
  events = utils.layers.MergeDeconvolution()(events)

  return events

def buildModel(nPixels=64, kernelSize=2, nConvolutions=9, lr=5.e-5, lossWeights=[100,10,1], kernel_regularisation=100., nGPUs=2):

  nLevels = math.log(nPixels, kernelSize) - 1
  m=nLevels%1
  if m > 1.e-5 and m < 1.-1.e-5:
    raise RuntimeError("The pixel array size, N, does not have the kernel size, k, as radix.  Require N=k^n, have N=" + str(nPixels) + ", k=" + str(kernelSize))

  nLevels = int(nLevels + 1.e-5)
  
  input = Input(shape=(nPixels, nPixels, 1, ))

  conv=[]
  deconv = []

  for i in range(nConvolutions):
    conv.append(Conv2D(filters=1, kernel_size=kernelSize, padding='same', kernel_initializer='glorot_normal', use_bias=False, kernel_regularizer=utils.regularisers.energyConservation(kernel_regularisation)))
    deconv.append(Conv2DTranspose(filters=1, kernel_size=kernelSize, strides=kernelSize, padding='same', kernel_initializer='glorot_normal', use_bias=False))

  shrink = MaxPooling2D(pool_size=kernelSize)

  filterMasks = []
  events = input

  for i in range(nLevels):
    events, mask = applyConvolutions(events, conv)
    events = shrink(events)
    filterMasks.append(mask)

  for i in range(nLevels):
    events = applyDeconvolutions(events, deconv, filterMasks[-1-i])

  if nGPUs > 0:
    with tf.device('/cpu:0'):
      cpumodel = Model(inputs=input, outputs=events)
    model = multi_gpu_model(cpumodel, gpus=nGPUs)
  else:
    model = Model(inputs=input, outputs=events)


  model.compile(loss=[utils.losses.activeLoss(lossWeights)], optimizer=Nadam(lr=lr))
  if nGPUs > 0:
    model.get_layer('model_1').summary()
  else:
    model.summary()
  return model





