import sys
import os
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


current_path = os.getcwd()
dataset = 'psm'
#datapath_folder = "/"
series_dict = {'smd': 28,
               'smap': 55,
               'psm': 1}

def read_data(dataset = 'psm', current_path = 'data/'):

    scalers = [MinMaxScaler() for i in range(series_dict[dataset])]

    train_path = current_path + '/' + dataset + '/train'
    file_names = os.listdir(train_path)
    file_names.sort()

    data = []
    data_all = []
    data_starts = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        with open(train_path + '/' + file_name) as f:
            this_data = pd.read_csv(train_path + '/' + file_name)
            this_data.drop(columns=[r'timestamp_(min)'], inplace=True)

    data_length = this_data.values.shape[0]
    full_train = this_data.values.astype(np.float32)
    full_train = np.nan_to_num(full_train)
    full_train = scalers[0].fit_transform(full_train)

    test_path = current_path + '/' + dataset + '/test'
    file_names = os.listdir(test_path)
    file_names.sort()
    test_data = []
    test_data_all = []
    test_data_starts = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        with open(test_path + '/' + file_name) as f:
            this_data = pd.read_csv(test_path + '/' + file_name)
            this_data.drop(columns=[r'timestamp_(min)'], inplace=True)

    test_data = this_data.values.astype(np.float32)
    test_data = np.nan_to_num(test_data)
    test_data = scalers[0].transform(test_data)

    test_target_path = current_path + '/' + dataset + '/test_label'
    file_names = os.listdir(test_target_path)
    file_names.sort()
    target = []
    for file_name in file_names:
        with open(test_target_path + '/' + file_name) as f:
            target_csv = pd.read_csv(test_target_path + '/' + file_name)
            target_csv.drop(columns=[r'timestamp_(min)'], inplace=True)
            target = target_csv.values
            target = target.astype(np.float32)

    return full_train, test_data, target
