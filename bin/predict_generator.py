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

eventCounter=0

def writeHeader(w, output):
  
  global eventCounter
    
  line = "E "+ str(eventCounter) + " -1 -1.0 1.0 1.0 0 -1 1 10001 10002 0 1 " + str(w) + "\n"
  output.write(line)
  output.write('N 1 "0" \n')
  output.write("U GEV MM\n")
  output.write("C 1. 1.\n")
  output.write("F 0 0 0. 0. 0. 0. 0. 0 0\n")
    
  eventCounter+=1
    
  return

def toMomentum(p):
  
  pT = p[2]
  pT2 = pT*pT
  cosphi = np.cos(p[1])
  sinphi = np.sin(p[1])
  tanhy = np.tanh(p[0])
  
  px = pT*cosphi
  py = pT*sinphi
  
  # Potentially add mass term here according to grid size?
  
  e2 = pT2 / (1. - tanhy*tanhy)
  
  pz2 = e2 - pT2
  
  return [px, py, np.sign(p[0])*np.sqrt(pz2), np.sqrt(e2)]

args = parser.parse_args()

np.random.seed(args.randomSeed)

output=open(args.output, "w")
output.write("\n")
output.write("HepMC::Version 2.06.09\n")
output.write("HepMC::IO_GenEvent-START_EVENT_LISTING\n")

epsilon = 0.1
nBins = args.pixels
binWidth = 2.*np.pi / float(nBins)

model = modelBuilder.buildModel(nPixels=args.pixels, lr=2.e-5, kernel_regularisation=50., kernelSize=args.kernel, nConvolutions=args.convolutions, merge_shower=True)

print "loading weights from " + args.weights
model.load_weights(args.weights)

gen = utils.loadData.generate_events(args.inputFiles, nEvents=args.nEvents, nPixels=args.pixels, normalise=False, predict_mode=True)

print "Generating " + str(gen.nEvents) + " events"

printCounter=1

for batch in range(gen.nBatches):
  input = gen.__getitem__(batch)
  weights = gen.getWeights(batch)
  prediction = model.predict(input).squeeze(axis=3)

  for ii in range(gen.batch_size):
    
    if eventCounter%printCounter ==0:
      print "Generated " + str(eventCounter) + " events"
    
      if printCounter != 10000 and eventCounter == 10* printCounter:
        printCounter *=10
    
    event = prediction[ii]
    weight = weights[ii]
    
    writeHeader(weight, output)

    particles = []
    yNominal = -np.pi -0.5*binWidth

    for ybin in range(0, nBins):
      yNominal += binWidth
      phiNominal = -0.5*binWidth

      for phibin in range(0, nBins):
        phiNominal += binWidth
        pT = event[ybin, phibin]

        if pT > epsilon:
          particle = [np.random.uniform(-0.5, 0.5)*binWidth + yNominal, np.random.uniform(-0.5, 0.5)*binWidth + phiNominal, pT]
          particles.append(particle)
#output.write(str(particle[0]) + " " + str(particle[1]) + " " + str(particle[2]) + "\n")

    output.write("V -1 0 0 0 0 0 2 "+str(len(particles)) + " 0\n")
    output.write("P 10001 2212 0 0 6.499999932280e+03 6.500000000000e+03 9.382720033633e-01 2 0 0 -1 0\n")
    output.write("P 10002 2212 0 0 -6.499999932280e+03 6.500000000000e+03 9.382720033633e-01 2 0 0 -1 0\n")
    barcode = 10003
    for p in particles:
      m = toMomentum(p)
      output.write("P " + str(barcode) + " 21 " + str(m[0]) + " " + str(m[1]) + " " + str(m[2]) + " " + str(m[3]) + " 0.0 1 0 0 -1 0\n"  )
      barcode += 1

output.write("HepMC::IO_GenEvent-END_EVENT_LISTING\n")








