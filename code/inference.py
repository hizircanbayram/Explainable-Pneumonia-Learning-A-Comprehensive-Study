from tqdm import tqdm
import numpy as np
from torchvision import transforms,models
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import os
from train import train_function, save_checkpoint
from test import test_function
from pneumonia import Pneumonia
import pandas as pd
from train import calculateMetrics

class_to_idx = {'Normal': 0, 'Lung Opacity': 1}
cat_to_name = {class_to_idx[i]: i for i in list(class_to_idx.keys())}
'''    
model = models.densenet121(pretrained=True) 
#model = models.resnet50(pretrained=True)
#model = models.googlenet(pretrained=True)
model.classifier = nn.Sequential(OrderedDict([
    ('fcl1', nn.Linear(1024,256)),
    ('dp1', nn.Dropout(0.3)),
    ('r1', nn.ReLU()),
    ('fcl2', nn.Linear(256,32)),
    ('dp2', nn.Dropout(0.3)),
    ('r2', nn.ReLU()),
    ('fcl3', nn.Linear(32,1)),
    #('out', nn.Softmax(dim=1)),
]))
'''
device = torch.device('cuda:0') 
batch_size = 16
train_on_gpu = True
checkpoint = torch.load('/home/dcl/Desktop/TrustableAIProject/dense_fold0/checkpoint_4.pth.tar', map_location={'cuda:2': 'cuda:0'}) 
model = checkpoint['model']

data_transforms = {
            'train': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(), # randomly flip and rotate
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]),
    
            'test': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ]),
    
            'valid': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            }


valid_data = Pneumonia('X_test_fold_0.txt', class_to_idx=class_to_idx, transforms=data_transforms['train'])
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=0, shuffle=True)

model.eval()
number_correct, number_data = 0, 0
all_preds = []
all_gts = []
for data, target in tqdm(valid_loader):
    if train_on_gpu:
        data, target = data.to(device), target.to(device)
    output = torch.squeeze(model(data))
    ############# calculate the accurecy
    #_, pred = torch.max(output, 1) 
    pred = output
    correct_tensor = pred.eq(target.data.view_as(pred))   
    all_preds = all_preds + pred.detach().cpu().numpy().tolist()
    all_gts = all_gts + target.data.tolist()
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                            else np.squeeze(correct_tensor.cpu().numpy())
    number_correct += sum(correct)
    number_data += correct.shape[0]
calculateMetrics(all_gts, all_preds, -1)
