#coding:utf-8
#general method to train a net
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from util import TrainDataset,ValDataset,train_model,initialize_model,set_parameter_requires_grad
from torchvision import datasets, models, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import time
import os,yaml,argparse
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# Flag for feature extracting. When False, we finetune the whole model,

def _init_():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', type=str, help='path to config file')
    args = parser.parse_args()

    cfgs = open(args.inputFile).read()
    cfgs = yaml.load(cfgs)
    print(cfgs)
    return cfgs


# Initialize the model for this run

def main():
    cfgs = _init_()
    cfgs = cfgs['Train']
    model_name= cfgs['model_name']
    feature_extract = cfgs['feature_extract']
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    if cfgs['loadPt']:
        checkpoint = torch.load(cfgs['loadPt'])
        state_dict = {}
        for key in checkpoint['model_state_dict'].keys():
            state_dict[key[7:]]=checkpoint['model_state_dict'][key]
        model_ft.load_state_dict(state_dict)
        start_epoch = checkpoint['epoch']
    print(model_ft)


    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfgs['mean'], cfgs['std'])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(cfgs['mean'], cfgs['std'])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    #image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    image_datasets = {}
    image_datasets ['train'] = TrainDataset(list_file = 'blademaster-train.lst',classes = classes,transform=data_transforms['train'])

    image_datasets ['val'] =TrainDataset(list_file = 'blade-val.lst',classes = classes,transform = data_transforms['val'])
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)


    # Send the model to GPU
    model_ft = model_ft.to(device)
    print(device)
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=cfgs['lr'], momentum=cfgs['momentum'])


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,start_epoch=cfgs['start_epoch'], num_epochs =cfgs['epoch'], is_inception=(model_name=="inception"))

    torch.save({
                'epoch': hist.index(max(hist))+1,
                'model_state_dict': model_ft.state_dict(),
                'optimizer_state_dict': optimizer_ft.state_dict()

                }, '%s-best.pt'%cfgs['savePath'])


if __name__ =="__main__":
    main()
    