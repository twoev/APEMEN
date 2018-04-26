import keras
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer


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



