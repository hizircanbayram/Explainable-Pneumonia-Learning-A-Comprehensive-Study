import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pydicom import dcmread
import pydicom as dcm

import os
import pandas as pd

from matplotlib import pyplot as plt
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSNetAug(nn.Module):
    def __init__(self, n_class=1, dropout_prob=.35, activation="sigmoid"):
        super(LSNetAug, self).__init__()
        self.n_class = n_class
        self.activation = activation
        self.intermediate_outputs = []
        conc_layers = [26, 17, 8, 3]
        
        # Get the features part of the VGG19 except for last MaxPooling Layer 
        self.vgg19 = models.vgg19(pretrained=True).features[:-1]
        
        for x in conc_layers:
            self.vgg19[x].register_forward_hook(self.hook)
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.upsample2d = nn.UpsamplingNearest2d(scale_factor=(2, 2))
           
        self.conv_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.conv_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
        self.batch_5 = nn.BatchNorm2d(512)
        
        self.conv_4_1 = nn.Conv2d(in_channels=1024,out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv_4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        self.batch_4 = nn.BatchNorm2d(256)
        
        self.conv_3_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        self.batch_3 = nn.BatchNorm2d(128)
        
        self.conv_2_1 = nn.Conv2d(in_channels=256, out_channels=64,  kernel_size=(3, 3), padding=1)
        self.conv_2_2 = nn.Conv2d(in_channels=64,  out_channels=64,  kernel_size=(3, 3), padding=1)
        self.batch_2 = nn.BatchNorm2d(64)
        
        self.conv_1_1 = nn.Conv2d(in_channels=128, out_channels=32,  kernel_size=(3, 3), padding=1)
        self.conv_1_2 = nn.Conv2d(in_channels=32,  out_channels=32,  kernel_size=(3, 3), padding=1)

        self.conv_out = nn.Conv2d(in_channels=32,  out_channels=self.n_class,  kernel_size=(3, 3), padding=1)

    def hook(self, module, inp, out):
        self.intermediate_outputs.append(out)
        
    def forward(self, x):      
        x = self.vgg19(x)
        self.intermediate_outputs.reverse()

        ####### CONV_5 ####### 
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = self.batch_5(x)
        x = self.upsample2d(x)
        
        ####### CONV_4 #######
        x = torch.cat([self.intermediate_outputs[0], x], dim=1)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = self.batch_4(x)
        x = self.upsample2d(x)
        
        ####### CONV_3 #######
        x = torch.cat([self.intermediate_outputs[1], x], dim=1)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = self.batch_3(x)
        x = self.upsample2d(self.dropout(x))
        
        ####### CONV_2 ####### 
        x = torch.cat([self.intermediate_outputs[2], x], dim=1)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = self.batch_2(x)
        x = self.upsample2d(self.dropout(x))
        
        ####### CONV_1 #######
        x = torch.cat([self.intermediate_outputs[3], x], dim=1)
        x = F.relu(self.conv_1_1(x))
        x = self.dropout(x)
        x = F.relu(self.conv_1_2(x))
        
        ####### OUTPUT #######
        x = torch.sigmoid(self.conv_out(x))
        
        # Clean the intermediate_outputs
        self.intermediate_outputs.clear()
        
        return x

    
    
class LSNetAugInference:
    def __init__(self, model_path, mean=None, std=None):
        """
        Instantiate LSNetAug object and load the LSNet-Aug lung segmentation model and its trained weights.

        :param model_path: Trained model path
        :param mean: Mean of train images
        :param std: Standard deviation of train images
        :return: LSNetAug object
        """

        self._mean = mean
        self._std = std
        self._model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load the model.

        :param model_path: Trained model path
        :return: Loaded model
        """
                
        model = LSNetAug().to(DEVICE)
        states = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(states)
        model.eval()
        
        return model

    def predict_segmentation_map(self, img, threshold=.5, convert_img=True):
        """
        Get lung segmentation map of given input image.

        :param img: Input image
        :param threshold: The decision threshold
        :param convert_img: If true, return black and white image.
        :return: Segmentation map
        """

        img = self.preprocess_img(img)
        img_batch = np.expand_dims(img, 0)

        img_batch = torch.from_numpy(img_batch).to(DEVICE)
        with torch.no_grad():
            result = self._model(img_batch).ge_(threshold).type(torch.uint8)
        
        if result.device == "cpu":
            result = result.numpy()
        else:
            result = result.cpu().numpy()

        if convert_img is True:
            result[np.where(result == 1)] = 255

        return result[0, 0]

    def preprocess_img(self, img):
        """
        Normalize input image with dataset mean and standart deviation and resize 448x448
        :param img: Input image
        :return: Normalized image
        """

        if len(img.shape) == 2:
            img = np.concatenate([img[np.newaxis, :, :]]*3, axis=0)
        elif len(img.shape) == 3:
            img = np.rollaxis(img, -1, 0)

        if self._mean:
            img = (img-self._mean) / self._std
        else:
            img = img / 255.

        return img.astype("float32")
    
    def resize_cxr_img(self, im, desired_size=448):
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        padding_info = (top, bottom, left, right)

        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return new_im, padding_info
    
    def resize_original_size(self, img, orig_shape):
        #print(img.shape, orig_shape)
        resized_mask = cv2.resize(img, orig_shape[::-1], interpolation=cv2.INTER_NEAREST)
        return resized_mask

    def crop_img(self, img, padding_info):
        (top, bot, left, right) = padding_info
        h, w = img.shape
        return img[top:h-bot, left:w-right]    
    
    
    
def getBBLungImage(lsnet, rsna_example):
    resized_img, padding_info = lsnet.resize_cxr_img(rsna_example)
    pred_img = lsnet.predict_segmentation_map(resized_img)
    cropped_pred_img = lsnet.crop_img(pred_img, padding_info)
    resized_pred_img = lsnet.resize_original_size(cropped_pred_img, rsna_example.shape)

    contours = cv2.findContours(resized_pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    xes = []
    yes = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        xes.append(x)
        xes.append(x+w)
        yes.append(y)
        yes.append(y+h)

    final_mask = np.zeros((rsna_example.shape[0], rsna_example.shape[1]), dtype=np.uint8)
    final_mask[min(yes):max(yes),min(xes):max(xes)] = 255
    final_img = np.bitwise_and(final_mask, rsna_example)  
    return final_img
