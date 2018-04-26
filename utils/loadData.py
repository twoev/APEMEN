import numpy as np
from scipy.sparse import lil_matrix
import keras


def loadData(fileNames, nEvents=-1, nPixels=64, normalise=True):

    if len(fileNames) == 0:
        return None

    data = np.zeros((0, nPixels, nPixels))

    for file in fileNames:
        mm=np.memmap(file, mode="r", dtype=float)
        mm.resize((len(mm)/(nPixels*nPixels), nPixels, nPixels))
        data = np.append(data, mm, axis=0)

        if nEvents > 0 and len(data) >= nEvents:
            break

    if nEvents > 0 and nEvents < len(data):
        data = data[:nEvents, :, :]

    if normalise:
        esum = np.sum(data, axis=(1,2), keepdims=True)
        data = data / esum
        data = float(nPixels*nPixels)*data
        
    data = np.expand_dims(data, axis=3)

    return data
    
def makeArray(fileNames, nEvents=-1, nPixels=64, normalise=True):

  if len(fileNames) == 0:
    return None

  txt = []
  maxEvents = 0
  for f in fileNames:
    file=open(f, "r")
    txt.append(file.readlines())
    for line in txt[-1]:
      if line == "======\n":
        maxEvents +=1

      if nEvents > 0 and maxEvents >= nEvents:
        break

    if nEvents > 0 and maxEvents >= nEvents:
      break

  if nEvents > maxEvents or nEvents < 0:
    nEvents = maxEvents

  data = np.zeros((nEvents, nPixels, nPixels))

  count = -1
  for t in txt:
    if count == nEvents:
      break
    
    for line in t:
      if line == "======\n":
        count +=1
        continue

      if count == nEvents:
        break

      vals=line.split()
      if len(vals) == 1:
        continue

      if vals[0] == "1" or vals[1] == "1":
        continue

      ybin = int(nPixels * float(vals[0]))
      phibin = int(nPixels * float(vals[1]))
      pT = float(vals[2])
      data[count, ybin, phibin] += pT

  if normalise:
      esum = np.sum(data, axis=(1,2), keepdims=True)
      data = data / esum
      data = float(nPixels*nPixels)*data

  data = np.expand_dims(data, axis=3)

  return data

class generate_events(keras.utils.Sequence):

    def __init__(self, filenames, batch_size=256, nPixels=64, nEvents=-1, normalise=True, predict_mode=False):

        self.predict_mode=predict_mode

        self.events = []
        self.weights=[]
        printCounter=100
        for n in filenames:
          
          if len(self.events) == nEvents:
            break
          
          file = open(n, "r")
          txt = file.readlines()
          for line in txt:

            if line == "======\n":
              
              if len(self.events)%printCounter == 0:
                print "loaded " + str(len(self.events)) + " events"
                if printCounter != 100000 and len(self.events) == 10*printCounter:
                  printCounter *=10
              
              if normalise and len(self.events) != 0:
                self.events[-1] /= self.events[-1].sum()
                self.events[-1] *= float(nPixels *nPixels)
              
              if len(self.events) == nEvents:
                break
              self.events.append(lil_matrix((nPixels, nPixels), dtype='float32'))
              continue

            vals=line.split()
            
            if len(vals) == 1:
              
              if predict_mode:
                self.weights.append(float(vals[0]))
              
              continue
    
            if vals[0] == "1" or vals[1] =="1":
              continue
            
            ybin = int(nPixels * float(vals[0]))
            phibin = int(nPixels * float(vals[1]))
            self.events[-1][ybin, phibin] += float(vals[2])


        nEvents = len(self.events)
        self.nBatches = int(nEvents / batch_size)
        self.nEvents = self.nBatches * batch_size
        
        while len(self.events) != self.nEvents:
          del self.events[-1]
          if predict_mode:
            del self.weights[-1]

        self.batch_size=batch_size
        self.nPixels=nPixels

        
    def __len__(self):
        return self.nBatches

    def getWeights(self, index):
      return self.weights[index*self.batch_size: (index+1)*self.batch_size]

    def __getitem__(self, index):
        dense_data=[]
        for d in self.events[index*self.batch_size:(index+1)*self.batch_size]:
            dense_data.append(np.expand_dims(d.todense(), axis=0))

        x = np.expand_dims(np.concatenate(dense_data, axis=0), axis=3)
        
        if self.predict_mode:
          return x
        
        return x, x

        
