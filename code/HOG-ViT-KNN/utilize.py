# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:00:19 2025

@author: wcfda
"""
from backbone import extract_features

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
from einops import rearrange

# ----------------------------------------------------------------------------- Data preparation functions
def label_filter(battery_labels, kwargs, shuffle=False, seed=0):
    features = kwargs['features']
    data_dir = kwargs['data_dir']
    invalid_labels = []
    for label in battery_labels:
        #label = battery_labels[0]
        for feature in features:
            #feature = features[0]
            image_path = data_dir + f'x/{feature}/{label}.png'
            image = cv2.imread(image_path, 0)
            inverse_image = 255 - image.astype(int)
            if inverse_image.max() == 0:
                invalid_labels.append(label)
                break
    valid_labels = [label for label in battery_labels if label not in invalid_labels]
    if shuffle:
        random.seed(seed)
        random.shuffle(valid_labels)
    return valid_labels

def read_x_data(battery_labels, kwargs, extract_feature=True):
    features = kwargs['features']
    data_dir = kwargs['data_dir']
    backbones = kwargs['backbones']
    
    images = []
    for label in battery_labels:
        #label = battery_labels[0]
        feat_images = [] 
        for feature in features:
            #feature = features[0]
            image_path = data_dir + f'x/{feature}/{label}.png'
            image = cv2.imread(image_path, 0)
            inverse_image = 255 - image.astype(int)
            image = inverse_image.astype(np.uint8)
            feat_images.append(image)
        images.append(feat_images)
    
    if extract_feature:
        image_feat_list = []
        images = np.array(images)
        n_battery = images.shape[0]
        images = rearrange(images, 'n f h w -> (n f) h w')
        for backbone in backbones:
            #backbone = backbones[0]
            feat_vectors = extract_features(images, backbone)
            image_feat_list.append(feat_vectors)
        image_features = np.concatenate(image_feat_list, axis=-1)
        X_data = rearrange(image_features, '(n f) d -> n f d', n=n_battery)
    else:
        X_data = np.array(images).astype(float)
    return X_data

def read_y_data(battery_labels, kwargs):
    data_dir = kwargs['data_dir']
    y_data = []
    with open(data_dir + 'y/battery_degrade_info', 'rb') as path: 
        battery_degrade_info = pickle.load(path)
    for label in battery_labels:
        #slope = battery_degrade_info[label]['degradation slope']
        #y_data.append([-1 / (10 * slope)])
        lifetime = battery_degrade_info[label]['retire cycle index']
        y_data.append([lifetime])
    return np.array(y_data)

class TokenDataset(Dataset):
    def __init__(self, X_data, y_data, transform=None):
        # X_data: (N, p, d), y_data: (N, )
        self.X = np.array(X_data).astype(np.float32)
        self.y = np.array(y_data).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        token = self.X[idx]
        lbl = self.y[idx]
        if self.transform:
            token = self.transform(torch.from_numpy(token))
        return token, lbl

def prepare_dataloaders(X_data, y_data, batch_size=4, shuffle=True):
    ds = TokenDataset(X_data, y_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader