import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import multilabel_confusion_matrix as cm
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import butter, lfilter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

## Function to get Individual Subjects's Data: 
def getSubjectsData(test_series=3,Test=False):

  # define which series will be used as the test series
  testSeries = test_series
  trainSeries = list(np.arange(1,9))
  trainSeries.remove(testSeries)

  
  path = '/content/drive/My Drive/Colab Notebooks/Bionic AI/Kaggle EEG Data/train/'

  subjects_data = []
  subjects_events = []


  if Test == False:
    for i in range(1,13):
  # initiate the dartaframes by passing it the first series of subject i
      stackData = pd.read_csv(path + f'subj{i}_series1_data.csv').iloc[:,1:]
      stackEvents = pd.read_csv(path + f'subj{i}_series1_events.csv').iloc[:,1:]
  # for subject i, import data from all training series and stack them
      for j in trainSeries[1:]:
          data = pd.read_csv(path + f'subj{i}_series{j}_data.csv').iloc[:,1:]
          stackData = pd.concat([stackData,data])  
      
          events = pd.read_csv(path + f'subj{i}_series{j}_events.csv').iloc[:,1:]
          stackEvents = pd.concat([stackEvents,events])
  # normalize the stack of series data for subject i
      stackDataNorm = ((stackData - stackData.mean(axis=0))/stackData.std(axis=0))
    
      stackDataNorm = stackDataNorm.reset_index(drop=True)
      stackEvents = stackEvents.reset_index(drop=True)

      subjects_data.append(stackDataNorm)
      subjects_events.append(stackEvents)

    return subjects_data, subjects_events

  else:
    for i in range(1,13):
   # get the testing data for all subjects. AKA, the series that will be used for testing for each subject i 
      data = pd.read_csv(path + f'subj{i}_series{testSeries}_data.csv').iloc[:,1:]
      events = pd.read_csv(path + f'subj{i}_series{testSeries}_events.csv').iloc[:,1:]

      dataNorm = ((data - data.mean(axis=0))/data.std(axis=0))

      subjects_data.append(dataNorm)
      subjects_events.append(events)

    return subjects_data, subjects_events


## function to get a specfic series data for a subject
def getSeriesData(subject_num,series_num):
  path = '/content/drive/My Drive/Colab Notebooks/Bionic AI/Kaggle EEG Data/train/'
  subject = pd.read_csv(path + f'subj{subject_num}_series{series_num}_data.csv').iloc[:,1:]
  events = pd.read_csv(path + f'subj{subject_num}_series{series_num}_events.csv').iloc[:,1:]
  subject = (subject - subject.mean(axis=0)) / subject.std(axis=0)
  return events


## Visualizing All Trials
def graphAllTrials(subject_num,series_num):

    events = getSeriesData(subject_num,series_num)
    Motions = events.columns.tolist()
    timeFrames = []
    vals = []
    count = 0
    for M in Motions:
      # plot each motion separately
        tf = events[events[M] == 1].index.tolist()
        val = np.ones(len(tf))
      # adding 0.5 to each motion to avoid overlapping and for better visualization
        val += count
        vals = list(vals)
        count += 0.05
        timeFrames.append(tf)
        vals.append(val)

    plt.style.use('dark_background')
    for i in range(len(Motions)):   
      plt.rcParams["figure.figsize"] = (30,1)
      plt.scatter(timeFrames[i], vals[i])


## Visualizing a single trials
def graphOneTrial(subject_num,series_num):
  events = getSeriesData(subject_num,series_num)
  Motions = events.columns.tolist()
  timeFrames = []
  vals = []
  count = 0
  for M in Motions:
    tf = events[events[M] == 1].index.tolist()[:150]
    val = np.ones(len(tf))
    val += count
    vals = list(vals)
    count += 0.05
    timeFrames.append(tf)
    vals.append(val)
 
  traces = []
  for i in range(len(Motions)):
    trace = go.Scatter(
      x=timeFrames[i],
      y=vals[i],
      name=Motions[i])
    traces.append(trace)

  fig = make_subplots(specs=[[{"secondary_y": False}]])
  for trace in traces:
    fig.add_trace(trace)
  fig['layout'].update(height = 300, width = 1600, template='plotly_dark')
  fig.update_yaxes(visible=False)
  fig.show()


## Function to Get All Data
def getAllData():
# define which subject will be used for testing
  trainSeries = [1,2,4,5,6,7,8]
  testSeries = 3
  path = '/content/drive/My Drive/Colab Notebooks/Bionic AI/Kaggle EEG Data/train/'

  subjects_data = []
  subjects_events = []

  for i in range(1,13):
    print(f'reading subject {i} out of 12')
    #initiate the dataframe to hold data for each subject by passing the first series of data to it
    stackData = pd.read_csv(path + f'subj{i}_series1_data.csv').iloc[:,1:]
    stackEvents = pd.read_csv(path + f'subj{i}_series1_events.csv').iloc[:,1:]
    # stack the rest of the series below the first for each subject i
    for j in trainSeries:
        data = pd.read_csv(path + f'subj{i}_series{j}_data.csv').iloc[:,1:]
        stackData = pd.concat([stackData,data])  
      
        events = pd.read_csv(path + f'subj{i}_series{j}_events.csv').iloc[:,1:]
        stackEvents = pd.concat([stackEvents,events])

    subjects_data.append(stackData)
    subjects_events.append(stackEvents)
    # concatenate all subjects data
  print(f'concatenating all data')
  allData = pd.concat(subjects_data)
  allEvents = pd.concat(subjects_events)
    # normalize the final dataframe and reset indices     
  dataNorm = ((allData - allData.mean(axis=0))/allData.std(axis=0))
    
  dataNorm = dataNorm.reset_index(drop=True)
  allEvents = allEvents.reset_index(drop=True)

  return dataNorm, allEvents 



## Function to Get Truncated Data for a Given Motion
def truncFrame(data,events,event):

  motionIndex = events[events[event] == 1].index.to_numpy()

  trunFrameRows = len(data.iloc[motionIndex[0]-100:motionIndex[0+149]+50,:][::20])
  truncFrame = np.zeros([trunFrameRows,32])

  i = 0
  count = 0

  while i < len(motionIndex):
    frame = data.iloc[motionIndex[i]-100:motionIndex[i+149]+50,:].to_numpy()
    truncFrame += frame[::20]
    count += 1
    i += 150
  truncFrame = truncFrame/count
  return truncFrame



## Function to Get Plot of Truncated Data for a Given Motions
def getGraph(data,events,event,ave=False):
  
# plot figures for how readings change across all 12 subjects for 'HandStart' action
  plt.style.use('default')
  fig = plt.figure(12, figsize=[50,10])
  
  frame = truncFrame(data,events,event)
  if ave == True:
    frame = frame.mean(axis=1)

  xnew = np.linspace(0, len(frame), 300) # 300 represents number of points to make between T.min and T.max

  spl = make_interp_spline(range(len(frame)), frame, k=3)  # type: BSpline
  power_smooth = spl(xnew)

  plt.plot(xnew, power_smooth)
  plt.show()


## Get Batches for the CNN
def getBatch(data, events, num_samples, Test=False):
    
    num_features = 32 # number of electrodes
    window_size = 512 # window size (number of previous timesteps we want to consider when making predictions)
    
    index = random.randint(window_size, len(data) - 16 * num_samples) # choose  a starting index: a number bigger than 1024 (window size) and less than the number of indexes that will be used by the batch
    if Test == False:
        indexes = np.arange(index, index + 16*num_samples, 16)

    else:
        index = random.randint(window_size, len(data) - num_samples) # much smaller dataset so dont have to make the 16 step jump between single batches
        indexes = np.arange(index, index + num_samples)

    X = np.zeros((num_samples, num_features, window_size//2))
    
    b = 0
    for i in indexes:
        
        start = i - window_size if i - window_size > 0 else 0
        
        tmp = data.iloc[start:i,:]
        X[b,:,:] = tmp[::2].transpose()
        
        b += 1
    y = events[events.index.isin(indexes)]
    y = y.to_numpy()

    return torch.DoubleTensor(X), torch.DoubleTensor(y) 


## Function to Train Subject Data
from tqdm import tqdm
def train(model, Xtrain, ytrain, epochs, batch_size,verbos=1):
#model.train() tells your model that you are training the model. So effectively layers 
#like dropout, batchnorm etc. which behave different on the train and test procedures 
#know what is going on and hence can behave accordingly.
  optimizer = torch.optim.Adadelta(model.parameters(), lr=1, eps=1e-10)
  model.train()
  for epoch in range(epochs):
    total_loss = 0 # set loss = 0
    
    for i in tqdm(range(len(Xtrain)//batch_size)):

      optimizer.zero_grad()
      x, y = getBatch(Xtrain, ytrain, batch_size, Test=False)
      while y.shape[0] != batch_size:
        x, y = getBatch(Xtrain, ytrain, batch_size, Test=False)
      outputs = model(x)
      loss = F.binary_cross_entropy(outputs.reshape(-1),y.reshape(-1)) # flattens both
      loss.backward() # backward propagation
      total_loss += loss.item()
      optimizer.step() # update the weights using the selected optimizer function 
      
      print(f'\t epoch: {epoch}, iteration: {i}/{len(Xtrain)//batch_size}, loss: {total_loss}')
      total_loss = 0


## function to get predictions
def getPredictions(model,Xtest,ytest,window_size,batch_size):
  
  model.eval()

  p = []
  tru = []

  while window_size < len(Xtest):
    if window_size + batch_size > len(Xtest):
      batch_size = len(Xtest) - window_size
    x_test, y_test = getBatch(Xtest, ytest, 2000, Test=True)
    x_test = (x_test)

    preds = model(x_test)
    #preds = preds.squeeze(1)

    p.append(np.array(preds.data))
    tru.append(np.array(y_test.data))

    window_size += batch_size
  preds = p[0]
  for i in p[1:]:
    preds = np.vstack((preds,i))
  
  test = tru[0]
  for i in tru[1:]:
    test = np.vstack((test,i))
  return preds, test

# CNN model used for train and test:

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv1d(32, 64, kernel_size=3,padding=0,stride=1) # why 32 and 64 ?
    self.bn = nn.BatchNorm1d(64)
    self.pool = nn.MaxPool1d(2,stride=2) # max pooling over a 2x2 window
    self.dropout1 = nn.Dropout(0.5)
    self.linear1 = nn.Linear(8128, 2048) # ????? isnt it 256x32 = 8192??
    self.dropout2 = nn.Dropout(0.5)
    self.dropout3 = nn.Dropout(0.5)
    self.linear2 = nn.Linear(2048, 124)
    self.dropout4 = nn.Dropout(0.5)
    self.dropout5 = nn.Dropout(0.5)
    self.linear3 = nn.Linear(124,6) ## check the output

    self.conv = nn.Sequential(self.conv1, nn.ReLU(inplace = True), self.bn, self.pool, 
                             self.dropout1)
    
    self.net = nn.Sequential(self.linear1, nn.ReLU(inplace=True), 
                             self.dropout2, self.dropout3, self.linear2, nn.ReLU(inplace=True),self.dropout4, self.dropout5, self.linear3   )

  def forward(self, x):
    batch_size = x.size(0)
    x = self.conv(x)
    x = x.reshape(batch_size, -1) # If there is any situation that you don't know how many columns you want but are sure of the number of rows, then you can specify this with a -1.
    out = self.net(x) # try the way pytorch does it
        
    return torch.sigmoid(out)

class LR(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear1 = nn.Linear(8192, 124) # ????? isnt it 256x32 = 8192??
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.linear2 = nn.Linear(124, 6)

    self.reg = nn.Sequential(self.linear1, nn.ReLU(inplace=True), 
                             self.dropout1, self.dropout2, self.linear2)

  def forward(self, x):
    batch_size = x.size(0)
    x = x.reshape(batch_size, -1) # If there is any situation that you don't know how many columns you want but are sure of the number of rows, then you can specify this with a -1.
    out = self.reg(x) # try the way pytorch does it

    return torch.sigmoid(out)

## FUnctions for The Animation Demo

# Batch Function extracting only the first trial from the subjects test set
# batch

## Get Batches for the CNN
def getBatchV2(test_data,test_events):  
  num_features = 32 # number of electrodes
  window_size = 512 # window size (number of previous timesteps we want to consider when making predictions)
  # here we are predicting for trial 1 in this series
  num_samples = 3400 - 1000
 # much smaller dataset so dont have to make the 16 step jump between single batches
  indexes = np.arange(1000, 3400)

  X = np.zeros((num_samples, num_features, window_size//2))
    
  b = 0
  for i in indexes:
    start = i - window_size if i - window_size > 0 else 0    
    tmp = test_data.iloc[start:i,:]
    X[b,:,:] = tmp[::2].transpose()
        
    b += 1
  y = test_events[test_events.index.isin(indexes)]
  y = y.to_numpy()

  return torch.DoubleTensor(X), torch.DoubleTensor(y) 

## Getting the loss between the predictions and true events to oprimize the threshold cut off for each event
def getLoss(x1true,x2true,x1pred,x2pred):
  return sum([abs(x1true-x1pred),abs(x2true-x2pred)])

## Getting the threshold cutoff for each event in the trial
def getBestThresh(preds,test):
  # threshold range
  threshold = list(np.arange(0,1,0.001))
  bestTh = {}
  events = [0,1,2,3,4,5]
  eventNums = {0: 'HandStart', 1:'FirstDigitTouch', 2:'BothStartLoadPhase', 3:'LiftOff', 4:'Replace', 5:'BothReleased'}
  plt.style.use('dark_background')

  for event in events:  
    truMotion = np.where(test[:,event] == 1)[0]
    losses = {}
    for th in threshold:
      predMotion = np.where(preds[:,event] > th)[0]
      try:
        loss = getLoss(truMotion[0],truMotion[-1],predMotion[0],predMotion[-1])
        #overlap = len(set(truMotion) & set(predMotion))
        lenDiff = abs(len(truMotion) - len(predMotion))
        losses[th] = loss + lenDiff

      except IndexError:
        pass
    bestTh[event] = min(losses, key=losses.get)
  return bestTh