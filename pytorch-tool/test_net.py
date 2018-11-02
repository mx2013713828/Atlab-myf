from util import TestDataset,initialize_model,eval_model

import json
list_file = 
transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testset = TestDataset(list_file=list_file,
                      transform=transforms)

testloader = DataLoader(testset, batch_size=64,
                        shuffle=True, num_workers=0)


# model_name= 'densenet'
feature_extract = False
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
checkpoint = torch.load('densenet-blade-35.pt')
state_dict = {}
for key in checkpoint['model_state_dict'].keys():
    state_dict[key[7:]]=checkpoint['model_state_dict'][key]
model_ft.load_state_dict(state_dict) 

results = eval_model(model_ft,testloader)

json.dump(results,'output-log.json')