# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 22:22:03 2025

@author: wcfda
"""
from crossViT import crossViT
from utilize import label_filter, read_x_data, read_y_data, TokenDataset, prepare_dataloaders
from backbone import (HOG, _get_weights_enum_if_exists, get_input_size_from_weights_or_model,
                      NumpyGrayDataset, build_feature_extractor_from_torchvision_model, 
                      extract_image_features, extract_features)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------- Training/evaluation functions
def train_one_epoch(data_loader):
    model.train()
    running_loss = 0.0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(data_loader.dataset)

def eval_model(data_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            running_loss += criterion(pred, y).item() * x.size(0)
    return running_loss / len(data_loader.dataset)

def evaluate(y_test_pred, test_labels):
    ground_truths = []
    predictions = []
    for i, label in enumerate(test_labels):
        info = battery_degrade_info[label]
        #edge_cycle_index = info['edge cycle index']
        #edge_capa_rate = info['edge capacity rate']
        retire_cycle_index = info['retire cycle index']
        #retire_rate = info['retire capacity rate']
        battery_lifetime = retire_cycle_index
        #pred_lifetime = round(edge_cycle_index + 10 * (edge_capa_rate - retire_rate) * y_test_pred[i][0])
        pred_lifetime = y_test_pred[i][0]
        ground_truths.append(battery_lifetime)
        predictions.append(pred_lifetime)
    return np.array(ground_truths), np.array(predictions)

# ----------------------------------------------------------------------------- Phase division
phase = 1
fold = 20
# ----------------------------------------------------------------------------- NASA PCoE data
source = 'NASA_APCoE'
max_lifetime = 200
history_cycle = int(phase*(max_lifetime/fold))
features = ['end temperature', 'capacity', 'electrolyte resistance', 'charge-transfer resistance']
validate = False
n_epoch = 10000
final_model_save_time = '2026-03-17-15-18-30'
# ----------------------------------------------------------------------------- CALCE accelarated aging data
source = 'UMaryland_CALCE'
max_lifetime = 600
history_cycle = int(phase*(max_lifetime/fold))
features = ['Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)', 'Charge_Energy(Wh)', 'Discharge_Energy(Wh)']
validate = False
n_epoch = 2000
final_model_save_time = '2026-03-17-12-25-11'
# ----------------------------------------------------------------------------- MIT data
source = 'MIT'
max_lifetime = 2000
history_cycle = int(phase*(max_lifetime/fold))
features = ['internal_resistance', 'charge_capacity', 'discharge_capacity']
validate = True
n_epoch = 3000
final_model_save_time = '2026-03-17-18-00-08'
# ----------------------------------------------------------------------------- Directories
data_dir = f'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/image data/{source}/phase{phase}({history_cycle}-cycle)/'
save_dir_prefix = f'C:/Users/wcfda/Desktop/Study/Projects/BRULP/code/image regression/crossViT/results/{source}/phase{phase}({history_cycle}-cycle)/'

# ----------------------------------------------------------------------------- Backbone
backbones = ['HOG'] #feature dimension: 8100
backbones = ['VGG'] #feature dimension: 25088
backbones = ['MobileNet'] #feature dimension: 1280
backbones = ['EfficientNet'] #feature dimension: 1280
backbones = ['VGG', 'MobileNet', 'EfficientNet'] #feature dimension: 27648
backbone_name = '_'.join(backbones)

# ----------------------------------------------------------------------------- Set seed
RND = 0
np.random.seed(RND)
torch.manual_seed(RND)
random.seed(RND)

# ----------------------------------------------------------------------------- Data division
with open(data_dir + 'battery_labels', 'rb') as path:
    battery_labels = pickle.load(path)
with open(data_dir + 'y/battery_degrade_info', 'rb') as path: 
    battery_degrade_info = pickle.load(path)
kwargs = {
    'features': features,
    'data_dir': data_dir,
    'backbones': backbones
    }
train_labels, other_labels = train_test_split(battery_labels, test_size=0.2, random_state=RND)
train_labels = label_filter(train_labels, kwargs, shuffle=True, seed=RND)
other_labels = label_filter(other_labels, kwargs)
X_train = read_x_data(train_labels, kwargs)
y_train = read_y_data(train_labels, kwargs)
train_loader = prepare_dataloaders(X_train, y_train)

if validate:
    val_labels, test_labels = train_test_split(other_labels, test_size=0.75, random_state=RND)
    X_val = read_x_data(val_labels, kwargs)
    y_val = read_y_data(val_labels, kwargs)
    val_loader = prepare_dataloaders(X_val, y_val)
else:
    test_labels = other_labels
X_test = read_x_data(test_labels, kwargs)
y_test = read_y_data(test_labels, kwargs)
test_loader = prepare_dataloaders(X_test, y_test)

# ----------------------------------------------------------------------------- Creat model
m = 9 if 'HOG' in backbones else 16
N, p, d = X_train.shape #number of batteries, number of battery attributes, feature dimension
num_patches_large = p
patch_dim_large = d
large_dim = 128
num_patches_small = p * m
patch_dim_small = d // m
small_dim = 64

model = crossViT(num_patches_small, num_patches_large, patch_dim_small, patch_dim_large, small_dim, large_dim)
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

# ----------------------------------------------------------------------------- Train model
opt = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss = float('inf')
best_state = None
history = {'train_loss': [], 'val_loss': []}

for epoch in range(n_epoch):
    train_loss = train_one_epoch(train_loader)
    val_loss = eval_model(val_loader) if validate else train_loss
    print(f"Epoch {epoch + 1:02d}: Train MSE={train_loss:.6f}")
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = model.state_dict()

save_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_dir = save_dir_prefix + f'{backbone_name}({save_time})/'
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), save_dir + 'last.pth')
torch.save(best_state, save_dir + 'best.pth')
with open(save_dir + 'history.pkl', 'wb') as fp:
    pickle.dump(history, fp)

# ----------------------------------------------------------------------------- Evaluate model
#save_time = final_model_save_time
save_dir = save_dir_prefix + f'{backbone_name}({save_time})/'
model_state = torch.load(save_dir + 'last.pth', weights_only=True)
model.load_state_dict(model_state)
model.eval()

# Show training results
train_ds = TokenDataset(X_train, y_train)
y_train_pred = model(torch.from_numpy(train_ds.X))
GT_lifetimes, vit_predictions = evaluate(y_train_pred.detach().numpy(), train_labels)

pred_errors = abs(vit_predictions - GT_lifetimes)
mean_error = pred_errors.mean()
RMSE = np.sqrt(np.mean(pred_errors**2))
print(f"Mean error: {mean_error:.6f}")
print(f"RMSE: {RMSE:.6f}")

with open(save_dir + 'history.pkl', 'rb') as fp:
    history = pickle.load(fp)
plt.plot(history['train_loss'], label='trainning loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Show testing results
test_ds = TokenDataset(X_test, y_test)
y_test_pred = model(torch.from_numpy(test_ds.X))
GT_lifetimes, vit_predictions = evaluate(y_test_pred.detach().numpy(), test_labels)

# ----------------------------------------------------------------------------- kNN correction
k = 3
alpha = 9/max(history_cycle, 10)
knn_predictions = []
pred_lifetimes = []
train_images = read_x_data(train_labels, kwargs, extract_feature=False)
test_images = read_x_data(test_labels, kwargs, extract_feature=False)
for i, x_test in enumerate(test_images):
    SSE_list = []
    for x_train in train_images:
        SSE = np.sum(abs(x_test - x_train))
        SSE_list.append(SSE)
    knn_indices = np.array(SSE_list).argsort()[:k]
    knn_labels = np.array(train_labels)[knn_indices]
    knn_lifetimes = []
    for label in knn_labels:
        lifetime = battery_degrade_info[label]['retire cycle index']
        knn_lifetimes.append(lifetime)
    knn_lifetime_pred = round(np.mean(knn_lifetimes))
    knn_predictions.append(knn_lifetime_pred)
    
    vit_lifetime_pred = vit_predictions[i]
    output = (1 - alpha)*knn_lifetime_pred + alpha*vit_lifetime_pred
    pred_lifetimes.append(output)

# ----------------------------------------------------------------------------- Show results
pred_errors = abs(vit_predictions - GT_lifetimes)
mean_error = pred_errors.mean()
RMSE = np.sqrt(np.mean(pred_errors**2))
print('crossViT model')
print(f'MAE: {mean_error:.6f}')
print(f'RMSE: {RMSE:.6f}')

knn_predictions = np.array(knn_predictions)
pred_errors = abs(knn_predictions - GT_lifetimes)
mean_error = pred_errors.mean()
RMSE = np.sqrt(np.mean(pred_errors**2))
print('kNN model')
print(f'MAE: {mean_error:.6f}')
print(f'RMSE: {RMSE:.6f}')

pred_lifetimes = np.array(pred_lifetimes)
pred_errors = abs(pred_lifetimes - GT_lifetimes)
mean_error = pred_errors.mean()
RMSE = np.sqrt(np.mean(pred_errors**2))
print('Final model')
print(f"MAE: {mean_error:.6f}")
print(f'RMSE: {RMSE:.6f}')

# ----------------------------------------------------------------------------- Save results
pred_results = {}
pred_results['actual'] = GT_lifetimes
pred_results['vit'] = vit_predictions
pred_results['knn'] = knn_predictions
pred_results['final'] = pred_lifetimes
with open(save_dir + 'pred_results.pkl', 'wb') as fp:
    pickle.dump(pred_results, fp)