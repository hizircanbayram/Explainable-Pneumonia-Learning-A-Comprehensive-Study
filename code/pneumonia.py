import os
import pydicom
from PIL import Image
import torch


root = os.path.abspath(os.path.join(os.getcwd()))


def make_one_hot(label, C=3):
    return torch.eye(C)[label, :]

class Pneumonia(object):
    '''
    file_name: dcm img paths and their corresponding labels for any fold of train/test sections.
    pick out the 1st row(img_name) and last row(labels)
    read by each line to get images and labels
    '''
    def __init__(self, file_name, class_to_idx, transforms=None):      
        self.imgs = []
        self.transform = transforms
        self.class_to_idx = class_to_idx
          
        with open(root + '/dataset/' + file_name) as mfile:
            for line in mfile:
                if '0' in line.split(', ')[1]:
                    continue
                cur_line = line.rstrip().split(", ")
                self.imgs.append((cur_line[0][1:][1:-1], int(cur_line[1][:-1][1])-1))
   
    def __getitem__(self, index):
        img, label = self.imgs[index]
        dcm_file = pydicom.read_file(img)
        img_arr = dcm_file.pixel_array
        img = Image.fromarray(img_arr).convert('RGB') 
        #make one hot
        #label = make_one_hot(label)
        #label = label.long()

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
