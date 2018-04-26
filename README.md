# APEMEN
Autoencoding Parton Emitting Model Encoded in Networks

Author: James Monk
This needs Keras, numpy, scipy.  If you want to run on (multi) GPU, you need TensorFlow and (duh) a Nvidia GPU.

The model design ensures self-similarity and adds randomness via the FilterMask layer that is in utils/layers.  A paper will be along to explain more at some point

Create the model by

import utils.model as modelBuilder

model = modelBuilder.buildModel()



