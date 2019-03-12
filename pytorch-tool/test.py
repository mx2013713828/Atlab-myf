#coding:utf-8
#mayufeng 2018-12-12 18:28:22
#general method to eval a net
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from util import TestDataset,ValDataset,train_model,initialize_model,set_parameter_requires_grad,eval_model
from torchvision import datasets, models, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import time
import os,yaml,argparse
import copy
import json
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def _init_():
    parser = argparse.ArgumentPar ser()
    parser.add_argument('inputFile', type=str, help='path to config file')
    args = parser.parse_args()

    cfgs = open(args.inputFile).read()
    cfgs = yaml.load(cfgs)
    print(cfgs)
    return cfgs
def main():
    cfgs = _init_()
    cfgs = cfgs['Test']
    print(cfgs)
    list_file = cfgs['testFile']
    feature_extract = cfgs['feature_extract']
    transform=transforms.Compose([
                            transforms.Resize(256)
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(cfgs['std'], cfgs['mean'])])

    testset = TestDataset(list_file=list_file,
                          transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=cfgs['batch_size'],
                            shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_name= 'densenet'
    feature_extract = False


    model_ft, input_size = initialize_model(cfgs['model_name'], cfgs['num_classes'], feature_extract, use_pretrained=False)
    model_ft = model_ft.to(device)
    checkpoint = torch.load(cfgs['loadPt'])
    state_dict = {}
    for key in checkpoint['model_state_dict'].keys():
        state_dict[key[7:]]=checkpoint['model_state_dict'][key]
    model_ft.load_state_dict(state_dict) 

    results = eval_model(model=model_ft,testloader=testloader,device=device)

    with open(cfgs['saveFile'],'w') as f:
        json.dump(results,f,indent=2)
if __name__ =="__main__":
    main()
    