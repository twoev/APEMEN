from keras.optimizers import Nadam, Optimizer
from keras import backend as K

class Nadam_entropy(Nadam):

  def __init__(self, temperature=0.1, **kwargs):
    self.temperature = temperature
    super(Nadam_entropy, self).__init__(**kwargs)

  def get_gradients(self, loss, params):
    grads = K.gradients(loss, params)

    probs = grads
    for i in range(len(params)):
      grads[i] /= params[i] + K.epsilon()
    
    #probs = grads / (params + K.epsilon())
    probs = K.abs(probs)
    probs /= K.sum(K.flatten(probs)) + K.epsilon()
    Ts = -self.temperature*K.sum(K.flatten(probs * K.log(probs)))
    delta_s = K.gradients(Ts, params)

    for i in range(len(grads)):
      grads[i] = grads[i] + delta_s[i]

#    grads = grads + delta_s

    if hasattr(self, 'clipnorm') and self.clipnorm > 0:
      norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
      grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
    if hasattr(self, 'clipvalue') and self.clipvalue > 0:
      grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
    return grads


