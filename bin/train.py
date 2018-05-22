#!/usr/bin/env python

import importlib, os,sys
import numpy as np
import argparse
import math
from keras.callbacks import ModelCheckpoint

import utils.loadData
import utils.model as modelBuilder
import utils.callbacks

parser = argparse.ArgumentParser(description="Train NN on input data")
parser.add_argument('--inputFiles', '-i', type=str, nargs='*', default=["input.memmap"])
parser.add_argument('--model', '-m', type=str, default='kernel2')
parser.add_argument('--nEvents', '-n', type=int, default=-1)
parser.add_argument('--output', '-o', type=str, default="")
parser.add_argument('--randomSeed', '-r', type=int, default=140280)
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--weights', '-w', type=str, default='')
parser.add_argument('--validation', '-v', type=str, nargs='*', default=[])
parser.add_argument('--learning-rate', '-l', type=float, default=5.e-5)
parser.add_argument('--batch-size', '-b', type=int, default=256)
parser.add_argument('--pixels', '-p', type=int, default=64)
parser.add_argument('--convolutions', '-c', type=int, default=9)
parser.add_argument('--kernel', '-k', type=int, default=2)

args = parser.parse_args()

try:
    from ROOT import TFile, ROOT, gROOT, TGraph, TH1F, TH2F
except:
    sys.stderr.write("\nCould not find the ROOT python modules.")
    raise
  
np.random.seed(args.randomSeed)

model = modelBuilder.buildModel(nPixels=args.pixels, kernelSize=args.kernel, lr=args.learning_rate, kernel_regularisation=50., lossWeights=[100,10,1], nConvolutions=args.convolutions)

doWeights = False

if args.weights is not "":
  doWeights=True
  print "loading weights from " + args.weights
  model.load_weights(args.weights)

print "using " + str(len(args.inputFiles)) + " input files"

input = utils.loadData.loadData(args.inputFiles, args.nEvents, nPixels=args.pixels)

val_input = utils.loadData.loadData(args.validation, args.nEvents, nPixels=args.pixels)

modelName = args.model

bestName = "weights/" + modelName + "_latest.h5"
saveBest = ModelCheckpoint(bestName, monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True, period=1)
bitslog = utils.callbacks.BitsLogger(nConvs=args.convolutions)
entropylog = utils.callbacks.EntropyLogger()


history = model.fit(input, input, epochs=args.epochs, batch_size=2*args.batch_size, validation_data=(val_input, val_input), callbacks=[saveBest, bitslog, entropylog])

if args.output is not "":
    output = args.output
else:
    output = modelName + "_"+ str(len(input))
  
h5 = output + ".h5"
json = output + ".json"
weights = output + "_weights.h5"
  
from keras import backend as K

nLevels = int(math.log(args.pixels, args.kernel) - 1 + 1.e-5)

for l in range(1, nLevels+1):
  print str(K.eval(model.get_layer('model_1').get_layer("filter_mask_" + str(l)).filterProbs))
  
for i in range(1, args.convolutions + 1):
    kernel = model.get_layer('model_1').get_layer("conv2d_" + str(i)).get_weights()[0][:,:,0,0]
    print "conv kernel " + str(i) + " = " +str(kernel)
  
for i in range(1, args.convolutions + 1):
    kernel = model.get_layer('model_1').get_layer("conv2d_transpose_" + str(i)).get_weights()[0][:,:,0,0]
    print "deconv kernel " + str(i) + " = " + str(kernel)

model.save_weights(weights)

#print str(bits.bits_history)

oldLoss = None
oldLoss_v = None
oldBits = None
oldEntropy=None

if doWeights and os.path.isfile(output + "_history.root"):
    oldHistoryFile = TFile(output + "_history.root", "READ")
    oldLoss_gr = oldHistoryFile.Get("training_loss").Clone()
    oldLoss_gr.SetName("oldLoss")
    oldLoss = np.array(oldLoss_gr.GetY(), "d")
    oldLoss_v_gr = oldHistoryFile.Get("validation_loss").Clone()
    oldLoss_v_gr.SetName("oldVal_loss")
    oldLoss_v = np.array(oldLoss_v_gr.GetY(), "d")
    oldBits_gr = oldHistoryFile.Get("bits").Clone()
    oldBits_gr.SetName("bits")
    oldBits = np.array(oldBits_gr.GetY(), "d")
    oldEntropy_gr = oldHistoryFile.Get("entropy").Clone()
    oldEntropy_gr.SetName("entropy")
    oldEntropy = np.array(oldEntropy_gr.GetY(), "d")

historyFile = TFile(output + "_history.root", "RECREATE")
historyFile.cd()
loss = np.array(history.history["loss"], "d")
loss_v = np.array(history.history["val_loss"])
bits_ar = np.array(bitslog.bits_history, "d")
entropy_ar = np.array(entropylog.entropy_history, "d")

if oldLoss is not None:
    loss = np.concatenate([oldLoss, loss])

if oldLoss_v is not None:
    loss_v = np.concatenate([oldLoss_v, loss_v])

if oldBits is not None:
    bits_ar = np.concatenate([oldBits, bits_ar])

if oldEntropy is not None:
    entropy_ar = np.concatenate([oldEntropy, entropy_ar])


epochs = np.arange(len(loss), dtype="d")
loss_gr = TGraph(len(epochs), epochs, loss)
loss_gr.SetName("training_loss")
loss_gr.Write()
loss_v_gr = TGraph(len(epochs), epochs, loss_v)
loss_v_gr.SetName("validation_loss")
loss_v_gr.Write()

bits_gr = TGraph(len(epochs), epochs, bits_ar)
bits_gr.SetName("bits")
bits_gr.Write()

entropy_gr = TGraph(len(epochs), epochs, entropy_ar)
entropy_gr.SetName("entropy")
entropy_gr.Write()


