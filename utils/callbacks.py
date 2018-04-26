import keras
from keras import backend as K
from keras.callbacks import Callback

import numpy as np

class BitsLogger(Callback):

  def __init__(self, nConvs=9, **kwargs):
    self.norm = 1./np.log(float(nConvs))
    self.bits_history=[]
    self.filterLayers=[]
    super(BitsLogger, self).__init__(**kwargs)


  def on_train_begin(self, logs):
    layers = self.model.layers
    for l in layers:
      if l.name == 'model_1':
          layers=l.layers

    for l in layers:
      if "filter_mask" in l.name:
        self.filterLayers.append(l)
        
  def on_epoch_end(self, epoch, logs={}):

    bitsum=0.
    for l in self.filterLayers:
      weights=K.flatten(l.filterProbs)
      b=-self.norm*K.sum(weights*K.log(weights))
      bitsum += b

    print(' Activation bits: ' + str(K.eval(bitsum)))
    logs['activation_bits'] = K.eval(bitsum)
    self.bits_history.append(K.eval(bitsum))

class EntropyLogger(Callback):

  def __init__(self, **kwargs):

    self.entropy_history=[]
    self.filterLayers=[]
    self.constant = 0.5*np.log(2*np.pi) + 0.5
    self.hmin=0.
    self.hmax=0.
    self.norm=1.
    super(EntropyLogger, self).__init__(**kwargs)

  def on_train_begin(self, logs):
    layers = self.model.layers
    for l in layers:
      if l.name == 'model_1':
        layers=l.layers
        
    for l in layers:
      if "filter_mask" in l.name:
        self.filterLayers.append(l)

    nFilters = K.eval(K.shape(self.filterLayers[-1].filterProbs)[-1])

    r=np.random.uniform(size=(1000000, nFilters))
    sigma = np.std(r, axis=1)
    self.hmin = 1.05 * np.log(np.amin(sigma, axis=0))
    self.hmax = 0.95 * np.log(np.amax(sigma, axis=0))
    self.norm = 1. / (self.hmax - self.hmin)

  def on_epoch_end(self, epoch, logs={}):

    s=0.
    for l in self.filterLayers:
      weights = K.flatten(l.filterProbs)
      s += self.norm*(K.log(K.std(weights)) - self.hmin)

    print(' entropy: ' + str(K.eval(s)) )
    logs['entropy'] = K.eval(s)
    self.entropy_history.append(K.eval(s))



