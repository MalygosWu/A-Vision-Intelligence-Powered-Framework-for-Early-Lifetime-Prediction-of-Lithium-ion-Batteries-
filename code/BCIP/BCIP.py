# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:23:05 2025

@author: wcfda
"""
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# ----------------------------------------------------------------------------- Auxiliary functions
def scale_feature(x, feature, bounds):
    lb = bounds.loc['min', feature]
    ub = bounds.loc['max', feature]
    scaled_x = (x - lb)/(ub - lb)
    return scaled_x

def generate_x_sample(data, feature, feature_bounds, show=False, image_name=None):
    cycle_NOs = data['Cycle_Index'].values
    feature_values = data[feature].values
    scaled_feat_values = scale_feature(feature_values, feature, feature_bounds)
    x, y = cycle_NOs[:history_cycle], scaled_feat_values[:history_cycle]
    
    f, ax = plt.subplots(1, 1, figsize = (1.28, 1.28))
    ax.plot(x, y, color = 'black', linewidth = 6)
    ax.set_xlim(1, history_cycle)
    ax.set_ylim(0, 1)
    ax.axis('off')
    if show:
        plt.show()
    if image_name is not None:
        path = save_dir + f'x/{feature}/{image_name}.png'
        f.savefig(path, dpi = 100)
    plt.close(f)
    
def preprocess(RULP_data, capacity_columns, interpolate=False):
    data = RULP_data.copy()
    if not 'Cycle_Index' in RULP_data.columns:
        num_cycle = len(data)
        data['Cycle_Index'] = np.arange(num_cycle) + 1
    for C_column in capacity_columns:
        capacities = data[C_column].values
        data[C_column] = capacities/rated_capacity
    if interpolate:
        data.interpolate(method='linear', limit_direction='both', inplace=True)
    return data

# ----------------------------------------------------------------------------- phase division
phase = 1
fold = 20

# ----------------------------------------------------------------------------- UMaryland CALCE accelarated aging data
max_lifetime = 600
history_cycle = int(phase*(max_lifetime/fold))
rated_capacity = 3.36 #Ah
retire_rate = 0.8
loess_frac = 0.1

capacity_columns = ['Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']
features = ['Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)', 'Charge_Energy(Wh)', 'Discharge_Energy(Wh)']
data_dir = 'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/RULP data/UMaryland_CALCE/'
save_dir = f'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/image data/UMaryland_CALCE/phase{phase}({history_cycle}-cycle)/'

# ----------------------------------------------------------------------------- NASA PCoE data
max_lifetime = 200
history_cycle = int(phase*(max_lifetime/fold))
rated_capacity = 2 #Ah
retire_rate = 0.8
loess_frac = 0.1

capacity_columns = ['capacity']
features = ['end temperature', 'capacity', 'electrolyte resistance', 'charge-transfer resistance']
data_dir = 'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/RULP data/NASA_APCoE/'
save_dir = f'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/image data/NASA_APCoE/phase{phase}({history_cycle}-cycle)/'

# ----------------------------------------------------------------------------- MIT data
max_lifetime = 2000
history_cycle = int(phase*(max_lifetime/fold))
rated_capacity = 1.1 #Ah
retire_rate = 0.81
loess_frac = 0.1

capacity_columns = ['charge_capacity', 'discharge_capacity']
features = ['internal_resistance', 'charge_capacity', 'discharge_capacity', 'maximum_temperature']
data_dir = 'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/RULP data/MIT/'
save_dir = f'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/image data/MIT/phase{phase}({history_cycle}-cycle)/'

# ----------------------------------------------------------------------------- Screen datasets
battery_labels = []
battery_degrade_info = {}
dataset_names = os.listdir(data_dir)
for feature in features:
    os.makedirs(save_dir + f'x/{feature}/', exist_ok=True) 
os.makedirs(save_dir + 'y/', exist_ok=True)

for name in dataset_names:
    #name = dataset_names[0]
    battery_data = pd.read_csv(data_dir + name)
    num_cycle = len(battery_data)
    data_is_applicable = False
    if num_cycle > history_cycle:
        cycle_indices = np.arange(num_cycle) + 1
        for C_column in capacity_columns:
            capacities = battery_data[C_column].values
            capacity_rates = capacities/rated_capacity
            smooth_rates = lowess(capacity_rates, cycle_indices, frac=loess_frac, it=3, return_sorted=False)
            hist_rates = smooth_rates[:history_cycle]
            condition1 = min(hist_rates) >= retire_rate
            condition2 = min(smooth_rates) < retire_rate
            if np.all([condition1, condition2]):
                data_is_applicable = True
                break
        
    if data_is_applicable:
        battery_label, _ = name.split('.')
        battery_labels.append(battery_label)
        # generate y samples for the image regression task
        edge_cycle_index = history_cycle
        retired_cycles = cycle_indices[smooth_rates < retire_rate]
        retire_cycle_index = min(retired_cycles)
        edge_capa_rate = capacity_rates[cycle_indices == edge_cycle_index][0]
        slope = (edge_capa_rate - retire_rate)/(edge_cycle_index - retire_cycle_index)
        battery_degrade_info[battery_label] = {
            'edge cycle index': edge_cycle_index,
            'edge capacity rate': edge_capa_rate,
            'retire cycle index': retire_cycle_index, 
            'retire capacity rate': retire_rate,
            'degradation slope': slope
            }

with open(save_dir + 'battery_labels', 'wb') as path: 
    pickle.dump(battery_labels, path)
with open(save_dir + 'y/battery_degrade_info', 'wb') as path: 
    pickle.dump(battery_degrade_info, path)
print(f'{len(battery_labels)} battery datasets selected')

# ----------------------------------------------------------------------------- Set data bounds
with open(save_dir + 'battery_labels', 'rb') as path: 
    battery_labels = pickle.load(path)
for i, battery_label in enumerate(battery_labels):
    #battery_label = battery_labels[0]
    raw_data = pd.read_csv(data_dir + f'{battery_label}.csv')
    RULP_data = preprocess(raw_data, capacity_columns)
    cycle_indices = np.arange(len(RULP_data)) + 1
    feature_data = RULP_data[features]
    for col in features:
        feature_data.loc[:, col] = lowess(RULP_data[col], cycle_indices, frac=loess_frac, it=3, return_sorted=False)
    hist_data = feature_data[:history_cycle]
    if i == 0:
        lb_data = hist_data.agg(['min'])
        ub_data = hist_data.agg(['max'])
    else:
        lb_data = pd.concat([lb_data, hist_data.agg(['min'])], ignore_index=True)
        ub_data = pd.concat([ub_data, hist_data.agg(['max'])], ignore_index=True)

feature_lb = lb_data.agg(['min'])
feature_ub = ub_data.agg(['max'])
feature_bounds = pd.concat([feature_lb, feature_ub])
for C_column in capacity_columns:
    feature_bounds.loc['min', C_column] = retire_rate
    feature_bounds.loc['max', C_column] = 1
has_NA = feature_bounds.isnull().values.any()
if has_NA: raise ValueError('Missing bounds detected')

# ----------------------------------------------------------------------------- Visualize data
for battery_label in battery_labels:
    raw_data = pd.read_csv(data_dir + f'{battery_label}.csv')
    RULP_data = preprocess(raw_data, capacity_columns, interpolate=True)
    # generate x samples for the image regression task
    for feature in features:
        generate_x_sample(RULP_data, feature, feature_bounds, image_name = battery_label)