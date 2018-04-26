import keras
from keras import backend as K
import numpy as np

# This func can be passed to a conv2d layer to try to encourage energy conservation,
# i.e. that the sum of all weights in the kernel is one
# The multiplier determines how large a penalty is applied for kernels that deviate from 1
def energyConservation(multiplier):

  def regulariser(weight_matrix):
    diff = 1. - K.sum(weight_matrix)
    return K.in_train_phase(multiplier * K.square(diff), 0.)

  return regulariser

def uniformProbabilities(multiplier, nConvs):

  def regulariser(weight_matrix):
    return multiplier*K.sum(K.square(weight_matrix - 1./nConvs))/nConvs

  return regulariser

#  The network weights undergo something akin to a phase transition during training
#  The network changes from a chaotic state to a much more ordered state in the space
#  of a few training epochs.  Class IV cellular automata occur around such phase transitions
#  The weights of the FilterMask layer represent the probability with which a given kernel
#  is active in the shower.  The information content in these sets of probabilities is given
#  by the Shannon entropy, s=-\sum pln_n(p), where the base of the logarithm is number of probabilities.
#  s=0 corresponds to a perfectly ordered state, s=1 is chaotic.  Higher n tends towards higher s.
#
#  During training, the NN will tend to transition to a lower entropy state.
#  By adding a regularisation term that is -s, we make the phase transition to the ordered state
#  harder, and try to keep the system near the phase transition.
#  In some ways, this is like trying to create a supercooled fluid


def maximiseEntropy(multiplier, nConvs):

  norm = 1. / np.log(float(nConvs))

  def regulariser(weight_matrix):
    s = norm * K.sum(weight_matrix*K.log(weight_matrix))
    return multiplier * s

  return regulariser




