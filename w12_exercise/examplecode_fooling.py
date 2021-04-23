from __future__ import print_function, division

import torch
import torch.nn as nn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms,  utils
import matplotlib.pyplot as plt
import time
import os

import torch.utils.model_zoo as model_zoo

import PIL.Image



from getimagenetclasses import *

#preprocessing: https://pytorch.org/docs/master/torchvision/models.html
#transforms: https://pytorch.org/docs/master/torchvision/transforms.html
#grey images, best dealt before transform
# at first just smaller side to 224, then 224 random crop or centercrop(224)
#can do transforms yourself: PIL -> numpy -> your work -> PIL -> ToTensor()


def loadimage2tensor(nm):

    image = PIL.Image.open(nm).convert('RGB')
    image =  transforms.Resize(300)(image)
    image =  transforms.ToTensor()(image)
    image =transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    print(image.size())
    return image

def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    # undoes the image standardization preprocessing,
    # you need it for the out of bounds check
    # TODO

    return res

def saveimgfromtensor(inpdata):
    #creates a numpy array from tensor, with normalization undone
    #output value range: 0,255,
    #output shape: (c,h,w)
    # you need it to save tensors as images and not as tensors (e.g. using PIL.Image)

    #TODO

    return saveimg

def uintimgtotensor(img):

    # inverse function for of def saveimgfromtensor(inpdata), you need it to check whether the tensor recreated from the numpy is still adversarial

    tt=torch.tensor(img.astype(np.float32)/np.float32(255.)).type(torch.FloatTensor)
    #print(tt,tt.size())
    image =transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tt).unsqueeze(0)
    return image


def predictontensor(im,model,use_gpu,targetclass):

    model.train(False)


    if use_gpu:
        im = im.to('cuda:0')


    outputs = model(im)
    with torch.no_grad():
        _, preds = torch.max(outputs.data, 1)


    cls=get_classes()
    print('current predicted class',preds.item(),  cls[preds.item()], 'score for predicted class:', outputs.data[0,preds].item(),'score for target class:', outputs.data[0,targetclass].item(),)

    return im,preds,outputs



#displays diff of two images
def diffimgvis(tensor1,tensor2):
    t=(tensor1-tensor2).squeeze(0).numpy()
    minv=np.amin(t)
    maxv=np.amax(t)

    print('diff to orig',np.mean(np.abs(t)),np.mean(np.abs(t))*255.0)

    t=(t-minv)/ max(maxv-minv,0.00001)
    r=np.transpose(t,[1,2,0])
    print('r',r.shape)
    plt.imshow(r)
    plt.show()

#displays an image
def visimg2(res):
    print('res',res.shape)
    r=np.transpose(res,[1,2,0])
    print('r',r.shape)
    plt.imshow(r)
    plt.show()



def fool_model2(nm, model, use_gpu=False):

    targetclass= 5 #949
    stepsize=0.01

    model.train(False)

    im=loadimage2tensor(nm).unsqueeze(0)
    print(im.size())
    cls=get_classes()

    inputs=im

    #TODO run your attack here on inputs



    #################################################

    #check  inputs.data if it is still adversarial after discretization
    saveimg=saveimgfromtensor(inputs.data) # convert to numpy with values {0,1,2,...,255}
    tmpim=uintimgtotensor(saveimg) # back to tensor
    _,preds,_=predictontensor(tmpim, model, use_gpu,targetclass)


    if preds.item()!=targetclass:
        print('adversarial failed due to discretization!')
        exit()

    #plot differences
    diffimgvis(tmpim,im.data)
    # plot adversatial
    visimg2(saveimg)


    savename='./saveimg.png'
    PIL.Image.fromarray(np.transpose(saveimg,[1,2,0]), 'RGB').save(savename)


    oimg=saveimgfromtensor(im)
    savename2='./oimg.png'
    PIL.Image.fromarray(np.transpose(oimg,[1,2,0]), 'RGB').save(savename2)


    #validate prediction after loading
    im=loadimage2tensor(savename).unsqueeze(0)
    _,preds,outputs=predictontensor(im, model, use_gpu,targetclass)
    #simpleclsimage(savename, model, use_gpu)


###############################

def runstuff():

    use_gpu=False


    #model, choose whatever you like
    model = models.resnet18(pretrained=True)

    if use_gpu:
        model = model.to('cuda:0')

    nm='./mrshout2.jpg'
    fool_model2( nm,model, use_gpu=use_gpu)


if __name__=='__main__':

    runstuff()
