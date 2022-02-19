import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def get_bb(img_t):
    for rec in train_labels_df[train_labels_df['patientId'] == img_t].values.tolist():
        try:
            return (rec[1], rec[2], rec[3], rec[4])
        except:
            return (None)


def splitData(data_list0, data_list1, data_list2, n_splits=3):
    def getChunk(data_list, indexes):
        return_list = []
        for i in indexes:
            return_list.append(data_list[i])
        return return_list
    
    def writeFile(data, data_name):
        textfile = open(data_name, "w")
        for element in data:
            #print(element)
            textfile.write(str(element) + "\n")
        textfile.close()        
        
    kf = KFold(n_splits=n_splits)
    for fold_no, (train_index, test_index) in enumerate(kf.split(data_list0)): # doesn't matter if;
        # it is data_list0 or data_list1 or data_list0. because we use it only for getting indexes.
        #print("FOLD NO: ", fold_no, "TRAIN:", train_index, "TEST:", test_index)
        X_train_0 = getChunk(data_list0, train_index)
        X_test_0 = getChunk(data_list0, test_index)
        X_train_1 = getChunk(data_list1, train_index)
        X_test_1 = getChunk(data_list1, test_index)
        X_train_2 = getChunk(data_list2, train_index)
        X_test_2 = getChunk(data_list2, test_index)
        
        X_train = X_train_0 + X_train_1 + X_train_2
        X_test = X_test_0 + X_test_1 + X_test_2
        
        writeFile(X_train, "dataset/X_train_fold_" + str(fold_no) + ".txt")
        writeFile(X_test, "dataset/X_test_fold_" + str(fold_no) + ".txt")
            


root_path = '/home/dcl/Desktop/TrustableAIProject/'
TRAIN_SAMPLES = 26400
          
balanced_label_df = pd.read_csv(root_path + 'dataset/balanced_label.txt')
detailed_class_info_df = pd.read_csv(root_path + 'dataset/stage_2_detailed_class_info.csv')
train_labels_df = pd.read_csv(root_path + 'dataset/stage_2_train_labels.csv')
detailed_class_info_df = detailed_class_info_df.groupby('class').apply(lambda x: x.sample(TRAIN_SAMPLES//3).reset_index(drop=True))

#np.savetxt(r'/home/tianshu/pneumonia/dataset/balanced_label.txt', df.values, fmt='%s')
f = open(root_path + 'dataset/balanced_label.txt', 'w')
arr = detailed_class_info_df.values
new_line = ''
for index in tqdm(range(arr.shape[0])):
    line = arr[index]
    img = line[0]
    label = line[1]
    if label=='No Lung Opacity / Not Normal':
        label = '0'
    elif label=='Normal':
        label = '1'
    elif label=='Lung Opacity':
        label= '2'
    
    new_line = root_path + 'dataset/rsna-pneumonia-detection-challenge/stage_2_train_images/' + img + '.dcm' + ',' + label + '\n'
    f.write(new_line)

f.close()

info_0 = []
info_1 = []
info_2 = []

with open(root_path + 'dataset/balanced_label.txt') as file:
    for line in file:
        cur_line = line.rstrip().split(",")
        if cur_line[1] == '0':         
            info_0.append(cur_line)
        elif cur_line[1] == '1':
            info_1.append(cur_line)
        elif cur_line[1] == '2':
            info_2.append(cur_line)

splitData(info_0, info_1, info_2)



