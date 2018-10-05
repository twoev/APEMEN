import keras
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
import numpy as np


# On each stage of the deconvolution we should only use the output from a single filter for each pixel.
# The different pixels can use different filter output.
# This means we must mask the filter output based on which filter was
# used during the equivalent convolution step
#
# If this model were being used to generate showering randomly, then a random filter could be used, but using a
# random filter during training means that the input and output stages are likely to be different.
# The mask here ensures that the same filter is used during convolution and deconvolution

class FilterMask(Layer):
  
  def __init__(self, kernel_regularizer=None, **kwargs):
    self.uses_learning_phase = True
    super(FilterMask, self).__init__(**kwargs)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
  
  def build(self, input_shape):
    self.filterProbs = self.add_weight((1, 1, 1, input_shape[3],), initializer='one', name='{}_filterProbs'.format(self.name), trainable=False, regularizer=self.kernel_regularizer)
    super(FilterMask, self).build(input_shape)
  
  def compute_output_shape(self, input_shape):
    return input_shape
  
  def call(self, x):
    
    absx = K.abs(x)
    
    max = K.max(absx, axis=3, keepdims=True)
    output = absx - max + K.epsilon()
    output = 0.5*K.sign(output) + 0.5
    
    # The number of active splittings.  Should be exactly one for a true splitting
    nActive = K.sum(output, axis=3, keepdims=True)
    # This mask is 1 if there are 1 or 0 active splittings, 0 otherwise
    activeMask = 0.5*(K.sign(1. - nActive + K.epsilon())) + 0.5
    
    # This is 1 if there is exactly 1 active filter, 0 otherwise
    active = output * activeMask
    
    #This is now the number of times each filter has been active
    activeCounter = K.sum(active, axis=(0,1,2), keepdims=True)
    # normalise to probability
    activeCounter = activeCounter / K.sum(activeCounter)
    
    self.add_update([K.moving_average_update(self.filterProbs, activeCounter, 0.9)], x)
    
    output = output * K.random_uniform(shape=K.shape(output), minval=0., maxval=1.)*self.filterProbs
    
    max = K.max(output, axis=3, keepdims=True)
    output = output - max + K.epsilon()
    output = 0.5*K.sign(output) + 0.5
    
    return output

class MergeDeconvolution(Layer):
  
  def __init__(self, **kwargs):
    super(MergeDeconvolution, self).__init__(**kwargs)
  
  def build(self, input_shape):
    super(MergeDeconvolution, self).build(input_shape)
  
  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1], input_shape[2], 1)
  
  def call(self, x):
    return K.sum(x, axis=3, keepdims=True)

class MergeShower(Layer):

  def __init__(self, cutoff=20., kernelSize=2, **kwargs):
    self.kernelSize = kernelSize
    self.cutoff=cutoff
    self.window = K.variable(np.ones((kernelSize, kernelSize, 1, 1)))
    
    super(MergeShower, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    if len(input_shape) != 2:
      raise RuntimeError("MergeShower layer requries two input images to merge")

    if input_shape[0] != input_shape[1]:
      raise RuntimeError("MergeShower layer requries two equal sized images to merge")

    return input_shape[0]

#  return image where all pixels below the cutoff are set to zero
  def aboveCutoff(self, image):
    image = image - self.cutoff
    active = 0.5*(K.sign(image) + 1.)
    return 0.5*(image + K.abs(image)) + self.cutoff*active

  def call(self, input):
    # This is the part of the pre-shower image that is above the cutoff
    me = self.aboveCutoff(input[0])
    # This is the part of the post-shower image that is above the cutoff
    showered = self.aboveCutoff(input[1])
    # This is the soft part of the post-shower image.
    soft = input[1] - showered

    # This is one in each pixel where the shower has a hard emission
    showerMask = 0.5*(K.sign(showered - 1.) + 1.)
    # And this is one where the pre-shower has a hard emission
    meMask = 0.5*(K.sign(me - 1.) + 1.)

    # These count the number of hard emissions from the pre- and post-shower images in each window
    nShower = K.conv2d(showerMask, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
    nME = K.conv2d(meMask, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")

    # So this is one if the window does not contain a new emission, zero if it does
    emissionAllowed = K.resize_images( (0.5*(K.sign(nME - nShower + 0.5) + 1.)),height_factor=self.kernelSize, width_factor=self.kernelSize, data_format="channels_last")
    emissionForbidden = 1. - emissionAllowed

    # And these then are allowed emissions from the shower because they have not added a new hard emission
    shower_allowed = showered * emissionAllowed
    
    showerSum = K.conv2d(showered, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
    outputSum = K.conv2d(input[1], self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
    
    ratio = K.resize_images((showerSum / (outputSum + 1.e-5)), height_factor=self.kernelSize, width_factor=self.kernelSize, data_format="channels_last")
    
    # And these emissions must be vetoed and replaced with the ME emissions
    #vetoed = showered * emissionForbidden
    
    # Find the ME and soft contributions in the veto regions
    meInVeto = me * emissionForbidden
    #softInVeto = soft * emissionForbidden
    
    # Correct the ME emissions in the veto regions by subtracting the soft emissions, which are allowed
    #meSum   = K.conv2d(meInVeto, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
    #softSum = K.conv2d(softInVeto, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
    #ratio = K.resize_images((meSum - softSum) / (meSum + 1.e-5), height_factor=self.kernelSize, width_factor=self.kernelSize, data_format="channels_last")
    
    return shower_allowed + soft + meInVeto*ratio

#softSum = K.conv2d(soft, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
#   meSum = K.conv2d(input[0], self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")

#   ratio = K.resize_images((meSum - softSum) / (meSum + 1.e-5), height_factor=self.kernelSize, width_factor=self.kernelSize, data_format="channels_last")
#   me = self.aboveCutoff(input[0] * ratio)

#   return me + soft
    
#    meSum = K.conv2d(me, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")


    
    
    # This is 1 where there is a ME emission, zero everywhere else
 #   meActive = 0.5*(K.sign(me - 1.) + 1.)
    # And this one then counts the number of ME emissions in each k X k window
 #   meSum = K.conv2d(meActive, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
    
    # This then is one if the window contains 2 or more ME emissions, zero otherwise
    #  Which indicates there was a splitting at the previous level
 #   meSplitting = 0.5*(K.sign(meSum - 1.1)+1.)
    # And upscale it to the original size
 #   meSplitting = K.resize_images(meSplitting, height_factor=self.kernelSize, width_factor=self.kernelSize, data_format="channels_last")

    # This contains all the hard emissions from the shower that match the ME splittings
 #   shower_allowed = showered * meSplitting

    # And these shower emissions should be reclustered!
 #   to_recluster = showered - shower_allowed

    # We find the location of the reclustered emission by pooling over the window to find the value of the hardest emission, subtract that from all pixels in the window, then take the sign -> the +ve pixel is the reclustered location
 #   reclustered_position = 0.5*(K.sign(to_recluster + 1.e-5 - K.resize_images(K.pool2d(to_recluster, pool_size=(self.kernelSize, self.kernelSize), strides=(self.kernelSize, self.kernelSize), pool_mode="max", data_format="channels_last"), height_factor=self.kernelSize, width_factor=self.kernelSize, data_format="channels_last")) + 1.)

    # Sum over all emissions in the window to find the reclustered value
 #   reclustered = K.conv2d(to_recluster, self.window, strides=(self.kernelSize, self.kernelSize), data_format="channels_last")
    # Then up-scale and multiply by the location so that only one pixel in the window is non-zero
 #   reclustered = K.resize_images(reclustered, height_factor=self.kernelSize, width_factor=self.kernelSize, data_format="channels_last") * reclustered_position

    # The final mereged emissions are the soft part + the allowed shower + the reclustered shower
#return shower_allowed + reclustered + soft
#return me * K.sum(showered, axis=(1,2,3), keepdims=True) / (K.sum(me, axis=(1,2,3), keepdims=True) + 1.e-5) + soft
 #   return soft + shower_allowed + reclustered
#return soft + me



















