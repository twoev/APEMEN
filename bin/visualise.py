#!/usr/bin/env python

import importlib, os,sys
import numpy as np
import argparse
import h5py
from keras import backend as K
import utils.model as modelBuilder

parser = argparse.ArgumentParser(description="Dump model params")
parser.add_argument('--weights', '-w', type=str, default='')
parser.add_argument('--pixels', '-p', type=int, default=64)
parser.add_argument('--convolutions', '-c', type=int, default=9)
parser.add_argument('--kernel', '-k', type=int, default=2)
parser.add_argument('--filter', '-f', type=int, default=1)
parser.add_argument('--output', '-o', type=str, default="testRun.memmap")

args = parser.parse_args()

model = modelBuilder.buildModel(nPixels=args.pixels, kernelSize=args.kernel, nConvolutions=args.convolutions, nGPUs=0)
#model.load_weights(args.weights)

weightFile = h5py.File(args.weights, 'r')

#kernel = model.get_layer("conv2d_" + str(args.filter)).get_weights()

filterProbs = np.array(weightFile['model_1']['filter_mask_1']['filter_mask_1_filterProbs:0'][0,0,0,:])

order = np.argsort(filterProbs)

print order

filterNumber = order[-args.filter]+1

kernel = [weightFile['model_1']['conv2d_' + str(filterNumber)]["kernel:0"]]

for i in range(1, args.convolutions+1):
  model.get_layer("conv2d_" + str(i)).set_weights(kernel)

#kernel = model.get_layer("conv2d_transpose_" + str(args.filter)).get_weights()
kernel = [weightFile['model_1']['conv2d_transpose_' + str(filterNumber)]["kernel:0"]]

for i in range(1, args.convolutions+1):
  model.get_layer("conv2d_transpose_" + str(i)).set_weights(kernel)

output_img=[]

nRuns=100

for j in range(nRuns):
  print "run " + str(j)

  input_img = np.random.random((1, args.pixels, args.pixels, 1))
  output_img.append(input_img[:])

  for i in range(20):
    img = model.predict(output_img[-1])
    npanel = args.pixels / args.kernel
    panel = np.zeros((1, npanel, npanel, 1))
    for k in range(3):
      for l in range(3):
        panel = panel + img[:, k*npanel:(k+1)*npanel, l*npanel:(l+1)*npanel, :]
  
    output_img[-1] = np.zeros((1, args.pixels, args.pixels, 1))
    for k in range(3):
      for l in range(3):
        output_img[-1][:, k*npanel:(k+1)*npanel, l*npanel:(l+1)*npanel, :] = panel[:]
    
    output_img[-1] = 0.5*(output_img[-1] + np.abs(output_img[-1]))
    max=np.amax(output_img[-1])
    output_img[-1] = output_img[-1] / max

output = np.memmap(args.output, mode="w+", shape=output_img[0].shape, dtype=float)

output[:] = output_img[0][:]

for j in range(1, nRuns):

 output += output_img[j] / float(nRuns)

max = np.amax(output)
output /= max

output.flush()
