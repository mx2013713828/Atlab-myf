#coding:utf-8
#test_util.py
import os,time,copy
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models,utils
from PIL import Image
from torch import nn
import pretrainedmodels
from torch.optim.lr_scheduler import *
import pretrainedmodels.utils as utils

def eval_model(model,testloader,device,class_names = ['pulp','sexy','normal']):

    model.eval()
    with torch.no_grad():
        results = {}
        count = 0
        for inputs,image_names in testloader:
            
            tic = time.time()
            inputs = inputs.to(device)
            # labels = labels.to(device)
            
            outputs = model(inputs)
            scores = nn.Softmax(dim=1)(outputs)
#             print(scores.size(),inputs.size())
            _, preds = torch.max(outputs, 1)
            #write inference log to json file


            for j in range(inputs.size()[0]):
                Confidence = []
                result = {}
                predict_scores = scores[j]
#                 print(predict_scores.size())
                image_name = image_names[j]
                if image_name != 'error':
                    result['Top-1 Index'] = [int(preds[j])]
                    result['Class'] =class_names[preds[j]]
                    for m in range(len(class_names)):
                        Confidence.append(float(scores[j][m]))
    #                     print(float(scores[j][m]))
                    result['Confidence'] = Confidence
                    results[image_name] = result
                elif image_name =='error':
                    print('error img ,skipping')
#             print(results)
            toc = time.time()
            count = count + 1
            print("Batch [{}]:\t batch_time={:.3f}s".format(count,toc-tic))
    return results
                

            # for j in range(inputs.size()[0]):
            #     images_so_far += 1
            #     ax = plt.subplot(num_images//2, 2, images_so_far)
            #     ax.axis('off')
            #     ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            #     imshow(inputs.cpu().data[j])

            #     if images_so_far == num_images:
            #         model.train(mode=was_training)
            #         return
        # model.train(mode=was_training)

class TestDataset(Dataset):
    def __init__(self,list_file,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
        self.transform = transform
        self.list_file = list_file
        self.image_list = open(list_file).read().splitlines()


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
#         image_name = self.image_list[idx]

        

        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)

        try:
            image_name = self.image_list[idx]
            image = Image.open(image_name.split(' ')[0])
            image = image.convert('RGB')
            image_name = image_name.split('/')[-1]
        except:
            image_name = self.image_list[idx+1]
            image = Image.open(image_name.split(' ')[0])
            image = image.convert('RGB')
            image_name = 'error'
#         image_name = image_name.split('/')[-1]
        if self.transform:
            sample = self.transform(image)
        sample = (sample,image_name)

        return sample

class TrainDataset(Dataset):
    def __init__(self,list_file,classes=None,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
        self.transform = transform
#         self.transform_v2 = transform_v2
        self.list_file = list_file
        self.image_list = open(list_file).read().splitlines()
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        try:
            image_name = self.image_list[idx]
            image = Image.open(image_name.split(' ')[0])
        except:
            image_name = self.image_list[idx+1]
            image = Image.open(image_name.split(' ')[0])
        
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = image.convert('RGB')
        label = int(image_name.split(' ')[1])
        if self.transform:
            sample = self.transform(sample)
        if self.classes:
            label = self.classes[label]
        

        sample = (sample,label)

        return sample

class ValDataset(Dataset):
    def __init__(self,list_file,classes = None,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
        self.transform = transform
#         self.transform_v2 = transform_v2
        self.list_file = list_file
        self.image_list = open(list_file).read().splitlines()
        self.classes = classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        
        try:

            image_name = self.image_list[idx]
            image = Image.open(image_name.split(' ')[0])
        except:
            image_name = self.image_list[idx+1]
            image = Image.open(image_name.split(' ')[0])
        
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = image.convert('RGB')
        label = label(image_name.split(' ')[1])
        if self.transform:
            sample = self.transform(sample)
        if self.classes:
            label = self.classes[label]
        

        sample = (sample,label)

        return sample



def eval(model, dataloader, criterion, is_inception=False):
    for inputs,label in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _,preds = torch.max(outputs,1)
        


def train_model(model, dataloaders, criterion, optimizer,start_epoch, savePath,num_epochs=50, is_inception=False,device=None):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = MultiStepLR(optimizer,milestones = [10,30],gamma=0.1)
    print('start_epoch',start_epoch)
    for epoch in range(start_epoch+1,num_epochs):
        tep = time.time()
        scheduler.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count =1
            for inputs, labels in dataloaders[phase]:
                tbt =time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                tbt1 = time.time()
                batchtime = tbt1-tbt
                print("batch %s time %s"%(count,batchtime))
                count = count+1
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        time.sleep(10)
        print('save model for epoch %s'%epoch)
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()

            }, '%s-%s.pt'%(savePath,epoch))
#         model = nn.DataParallel(model)
#         model = model.to(device)
        print('save successful')          

    tep1 = time.time()
    tepoch =tep1-tep

    print('epoch time {}'.format(tepoch))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        all_parameters = []
        for name,param in model.named_parameters():
            all_parameters.append(name)
        for name,param in model.named_parameters():
            if name in all_parameters:
                param.requires_grad = True
            else:
                param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    elif model_name == 'se_resnext50_32x4d':
        "se_resnext 50 32x4d"
        if use_pretrained==True:
            model_ft = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
            dim_feats = model_ft.last_linear.in_features # =2048
            model_ft.last_linear = nn.Linear(dim_feats, num_classes)
        else:
            #to do
            model_ft = pretrainedmodels.__dict__[model_name](pretrained=None)
            dim_feats = model_ft.last_linear.in_features # =2048
            model_ft.last_linear = nn.Linear(dim_feats, num_classes)
        input_size = 224
    elif model_name == 'se_resnet50':
        if use_pretrained==True:
            model_ft = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
            dim_feats = model_ft.last_linear.in_features # =2048
            model_ft.last_linear = nn.Linear(dim_feats, num_classes)
        else:
            #to do
            model_ft = pretrainedmodels.__dict__[model_name](pretrained=None)
            dim_feats = model_ft.last_linear.in_features # =2048
            model_ft.last_linear = nn.Linear(dim_feats, num_classes)
        input_size=224
    else:
        if use_pretrained==True:
            model_ft = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
            dim_feats = model_ft.last_linear.in_features # =2048
            model_ft.last_linear = nn.Linear(dim_feats, num_classes)
        else:
            #to do
            model_ft = pretrainedmodels.__dict__[model_name](pretrained=None)
            dim_feats = model_ft.last_linear.in_features # =2048
            model_ft.last_linear = nn.Linear(dim_feats, num_classes)
        input_size=224
    return model_ft, input_size
