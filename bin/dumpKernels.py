#!/usr/bin/env python

import importlib, os,sys
import numpy as np
import argparse
from keras import backend as K
import utils.model as modelBuilder

import math

parser = argparse.ArgumentParser(description="Dump model params")
#parser.add_argument('--model', '-m', type=str, default='kernel2')
#parser.add_argument('--output', '-o', type=str, default="")
parser.add_argument('--weights', '-w', type=str, default='')
parser.add_argument('--pixels', '-p', type=int, default=64)
parser.add_argument('--convolutions', '-c', type=int, default=9)
parser.add_argument('--kernel', '-k', type=int, default=2)

args = parser.parse_args()

try:
  from ROOT import TFile, ROOT, gROOT, TGraph, TH1F, TH2F
except:
  sys.stderr.write("\nCould not find the ROOT python modules.")
  raise

model = modelBuilder.buildModel(nPixels=args.pixels, kernelSize=args.kernel, nConvolutions=args.convolutions)
model.load_weights(args.weights)

singleCore = model.get_layer('model_1')

nLevels = int(math.log(args.pixels, args.kernel) - 1 + 1.e-5)

for i in range(1, args.convolutions+1):
  kernel = singleCore.get_layer("conv2d_" + str(i)).get_weights()[0][:,:,0,0]
  print "conv kernel " + str(i) + " = " + str(kernel)

for i in range(1, nLevels+1):
  print str(K.eval(singleCore.get_layer("filter_mask_" + str(i)).filterProbs ))

for i in range(1, args.convolutions+1):
  kernel = singleCore.get_layer("conv2d_transpose_" + str(i)).get_weights()[0][:,:,0,0]
  print"deconv kernel " + str(i) + " = "+ str(kernel)
  





