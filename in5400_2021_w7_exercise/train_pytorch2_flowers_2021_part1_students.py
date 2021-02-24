from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os

#import skimage.io
import PIL.Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class dataset_flowers(Dataset):
    def __init__(self, root_dir, trvaltest, transform=None):


        self.root_dir = root_dir

        self.transform = transform
        self.imgfilenames=[]
        self.labels=[]


        if trvaltest==0:
            #load training data
            file = open("./102flowersn/flowers_data/trainfile.txt")
            path = "./102flowersn/flowers_data/jpg/"
            for i, line in enumerate(file):
                image = line.split()[0]
                flower = int(line.split()[1])
                self.imgfilenames.append(path + image)
                self.labels.append(flower)
            file.close()
        elif trvaltest==1:
            #load validation data
            file = open("./102flowersn/flowers_data/valfile.txt")
            path = "./102flowersn/flowers_data/jpg/"
            for i, line in enumerate(file):
                image = line.split()[0]
                flower = int(line.split()[1])
                self.imgfilenames.append(path + image)
                self.labels.append(flower)
            file.close()
        elif trvaltest==2:
            #load test data
            file = open("./102flowersn/flowers_data/testfile.txt")
            path = "./102flowersn/flowers_data/jpg/"
            for i, line in enumerate(file):
                image = line.split()[0]
                flower = int(line.split()[1])
                self.imgfilenames.append(path + image)
                self.labels.append(flower)
            file.close()
        #TODO
        else:
            #TODO: print some error + exit() or an exception
            raise ValueError('init error')


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        filename = self.imgfilenames[idx]
        image = PIL.Image.open(filename)
        label = self.labels[idx]

        sample = {'image': image, 'label': label, 'filename': filename}

        return sample



def train_epoch(model,  trainloader,  losscriterion, device, optimizer ):

    model.train() # IMPORTANT

    losses = list()
    for batch_idx, data in enumerate(trainloader):
        #TODO trains the model
        inputs=data[0].to(device)
        labels=data[1].to(device)


        optimizer.zero_grad()

        output = model(inputs)

        #Note may have to apply squeeze here
        loss = criterion(output.squeeze(1), labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def evaluate_acc(model, dataloader, losscriterion, device):

    model.eval() # IMPORTANT

    losses = []
    curcount = 0
    accuracy = 0

    with torch.no_grad():
        for ctr, data in enumerate(dataloader):
            #TODO
            #computes predictions on samples from the dataloader
            # computes accuracy, to count how many samples, you can just sum up labels.shape[0]

            print ('epoch at',len(dataloader.dataset), ctr)
            inputs = data[0].to(device)
            outputs = model(inputs)

            labels = data[1]
            labels = labels.float()
            cpuout= outputs.to('cpu')

            #_, preds = torch.max(cpuout, 1)
            preds = ( cpuout >= 0.5 ).squeeze(1)
            running_corrects += torch.sum(preds == labels.data)

        accuracy = running_corrects.double() / len(dataloader.dataset) # this does not work if one uses a datasampler!!!


    return accuracy.item() , np.mean(losses)


def train_model_nocv_sizes(dataloader_train, dataloader_test ,  model ,
                           losscriterion, optimizer, scheduler, num_epochs, device):

    best_measure = 0
    best_epoch =-1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train(True)
        losses=train_epoch(model,  dataloader_train,  losscriterion,  device , optimizer )

    if scheduler is not None:
        scheduler.step()

    model.train(False)
    measure, meanlosses = evaluate_acc(model, dataloader_test, losscriterion, device)
    print(' perfmeasure', measure)

    if measure > best_measure: #higher accuracy is better accuracy
        bestweights= model.state_dict()
        best_measure = measure
        best_epoch = epoch
        print('current best', measure, ' at epoch ', best_epoch)

    return best_epoch, best_measure, bestweights



def runstuff_finetunealllayers():

    #someparameters
    batchsize_tr=16
    batchsize_test=16
    maxnumepochs=2 #TODO can change when code runs here

    #TODO depends on what you can use
    device= torch.device('cpu') #torch.device('cuda')

    numcl=102
    #transforms
    data_transforms = {}
    data_transforms['train']=transforms.Compose([
          transforms.Resize(224),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    data_transforms['val']=transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

    #TODO

    datasets={}
    datasets['train']= dataset_flowers(None, 0)
    datasets['val']= dataset_flowers(None, 1)
    datasets['test']= dataset_flowers(None, 2)

    #TODO
    dataloaders={}
    dataloaders['train']= torch.utils.data.DataLoader(datasets['train'], batch_size = 8, shuffle=True)
    dataloaders['val']= torch.utils.data.DataLoader(datasets['val'], batch_size = 32, shuffle=False)
    dataloaders['test']=torch.utils.data.DataLoader(datasets['test'], batch_size = 32, shuffle=False)



    #model
    model = models.resnet18()

    model.to(device)

    #TODO
    criterion = torch.nn.CrossEntropyLoss()

    lrates=[0.01, 0.001]

    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = None
    for lr in lrates:

        #TODO
        optimizer = None

        best_epoch, best_perfmeasure, bestweights = train_model_nocv_sizes(dataloader_train = dataloaders['train'],
                                                                           dataloader_test = dataloaders['val'] ,
                                                                           model = model ,
                                                                           losscriterion = criterion ,
                                                                           optimizer = optimizer,
                                                                           scheduler = None,
                                                                           num_epochs = maxnumepochs ,
                                                                           device = device)

        if best_hyperparameter is None:
          best_hyperparameter = lr
          weights_chosen = bestweights
          bestmeasure = best_perfmeasure

        elif True:  #TODO what criterion here?
          if best_perfmeasure > bestmeasure:
              best_hyperparameter = lr
              weights_chosen = bestweights
          pass

    model.load_state_dict(weights_chosen)

    accuracy, testloss = evaluate_acc(model = model ,
                                    dataloader  = dataloaders['test'],
                                    losscriterion = criterion, device = device)

    print('accuracy val',bestmeasure , 'accuracy test',accuracy  )



def runstuff_finetunelastlayer():

  pass
  #TODO


def runstuff_fromscratch():

  pass
  #TODO




if __name__=='__main__':

  #runstuff_fromscratch()
  runstuff_finetunealllayers()
  #runstuff_finetunelastlayer()
