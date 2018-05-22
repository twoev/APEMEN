#!/usr/bin/env python

import importlib, os,sys
import numpy as np
import argparse
from keras.callbacks import ModelCheckpoint

import utils.loadData
import utils.model as modelBuilder

parser = argparse.ArgumentParser(description="Evaluate trained NN on input data")
parser.add_argument('--inputFiles', '-i', type=str, nargs='*', default=["input.memmap"])
parser.add_argument('--nEvents', '-n', type=int, default=-1)
parser.add_argument('--output', '-o', type=str, default="testRun")
parser.add_argument('--randomSeed', '-r', type=int, default=100217)
parser.add_argument('--weights', '-w', type=str, default='testRun')
parser.add_argument('--pixels', '-p', type=int, default=64)
parser.add_argument('--convolutions', '-c', type=int, default=9)
parser.add_argument('--kernel', '-k', type=int, default=2)


args = parser.parse_args()

try:
    from ROOT import TFile, ROOT, gROOT, TGraph, TH1F
except:
    sys.stderr.write("\nCould not find the ROOT python modules.")
    raise

np.random.seed(args.randomSeed)

model = modelBuilder.buildModel(nPixels=args.pixels, lr=2.e-5, kernel_regularisation=50., kernelSize=args.kernel, nConvolutions=args.convolutions)

print "loading weights from " + args.weights
model.load_weights(args.weights)

input = utils.loadData.loadData(args.inputFiles, args.nEvents, nPixels=args.pixels, normalise=False)

prediction = model.predict(input).squeeze(axis=3)

output = np.memmap(args.output, mode="w+", shape=prediction.shape, dtype=float)
output[:] = prediction[:]
output.flush()

