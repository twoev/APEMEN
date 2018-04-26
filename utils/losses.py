import keras
import numpy as np
from keras import backend as K

def gaussKernel(width):

  sigma = 0.5*float(width)
  x = np.arange(width) + 0.5
  x = x / sigma - 1.
  x = np.exp(-0.5*x*x)
  
  k=np.outer(x,x)
  norm = np.sum(k)
  return K.variable(np.expand_dims(np.expand_dims(k/norm, axis=2), axis=3))

# The input pixel arrays are sparse, with only a few % of pixels occupied.
# Standard MSE loss averages the MSE over all pixels, which causes the NN to
# learn mostly about the inactive pixels
# With this active loss, we treat the unoccupied target pixels as a single large pixel
# The target is used to define a mask, and the MSE between the target and prediction is
# computed only for those pixels that are not masked out.
# The sum of pixels outside the masked area is computed for the prediction and added to the loss.
# This means that any predicted activity outside target active pixels is treated the same,
# but a single large emission is treated equally to a large number of small emissions whose energy
# is equal to the larger emission.
# To avoid aliasing effects caused by emissions in the target and prediction in neighbouring pixels,
# we also allow both the target and prediction to be smeared by a n X n Gaussian kernel.
# This lowers the loss when the predicted emission is near a target emission.
# The loss is a weighted sum over different smear radii, and the weights can be specified.
# By changing the weights, the loss becomes more or less sensitive to wide angle effects

def activeLoss(weights):

  weights = np.array(weights, dtype=float) / np.sum(weights)
  kernels = []
  for i in range(1, len(weights)+1):
    kernels.append(gaussKernel(i))

  def loss(target, prediction):

    sum = 0.
    weightSum = 0.

    for i in range(len(weights)):

      x = K.conv2d(target, kernels[i], strides=(1,1), padding='same', data_format="channels_last")
      y = K.conv2d(prediction, kernels[i], strides=(1,1), padding='same', data_format="channels_last")

      active = 0.5*K.sign(x - 10.*K.epsilon()) + 0.5
      inactive = 1. - active

      nActive = K.sum(active, axis=(1,2)) + K.epsilon()
      nInactive = 64.*64.-nActive

      w = K.sum(K.square(x - active*y), axis=(1,2)) / nActive + K.square(K.sum(inactive*K.abs(y), axis=(1,2))) / nInactive

      sum += weights[i]*K.mean(0.5*w, axis=0)
      weightSum += weights[i]

    return sum / weightSum

  return loss



