import glob
import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch import optim

from .dataset import dataset_generator

num_patch = 4

def split_train_val(dataset, num_val=1):
    dataset = list(dataset)
    length = len(dataset)
    n = num_val
    random.shuffle(dataset)
    if num_val>0:
        return {'train': dataset[:-n], 'val': dataset[-n:]}
    else:
        return {'train': dataset}
    

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def train_nn(
    net,
    device,
    json_folder,
    label_name_to_value,
    epochs=5,
    batch_size=1,
    lr=0.1,
    num_channels=3
):
    # Create dataset
    ids = glob.glob(json_folder+'/*.json')
    ids = ((id, (i, j))  for id in ids for i in range(num_patch) for j in range(num_patch))
    iddataset = split_train_val(ids, num_val=1)
    
    # Specify the optimizer
#     optimizer = optim.Adam(net.parameters(),
#                           lr=lr,
#                           weight_decay=0.0005,
#                           amsgrad=False
#                           )
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    # Specify the loss function
    criterion = nn.BCELoss()
    loss_min = None
    best_epoch = 0
    loss_list = []
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        start_time = time.time()
        net.train()
        
        # Reset dataset generators
        train = dataset_generator(iddataset['train'], label_name_to_value, num_channels)
        val = dataset_generator(iddataset['val'], label_name_to_value, num_channels)
        
        # Start training
        epoch_loss = 0
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b]).astype(np.float32)
            
            imgs = torch.from_numpy(imgs).to(device)
            true_masks = torch.from_numpy(true_masks).to(device)
            
            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
        
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_epoch = epoch_loss / (i+1)
        print('Epoch finished ! Loss: {}'.format(loss_epoch))
        
        loss_list.append(loss_epoch)
        if loss_min is None:
            loss_min = loss_epoch
        if loss_epoch < loss_min:
            print('Update best epoch to {}'.format(epoch))
            loss_min = loss_epoch
            best_epoch = epoch
            best_model = net.state_dict()
    finish_time = time.time()
    print('Training finished in {}mins!'.format(round((finish_time-start_time)/60, 2)))
    
    final_model = net.state_dict()
    return loss_list, best_epoch, best_model, final_model


def train_nn_2(net, device, ds_loader, lr, epochs):
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    # Specify the loss function
    criterion = nn.BCELoss()
    loss_min = None
    best_epoch = 0
    loss_list = []
    start_time = time.time()
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # Start training
        epoch_loss = 0
        for i, sample_batch in enumerate(ds_loader):
            if device == 'cuda':
                imgs = sample_batch['image'].cuda()
                true_masks = sample_batch['mask'].cuda()
            else:
                imgs = sample_batch['image']
                true_masks = sample_batch['mask']      

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_epoch = epoch_loss / (i+1)
        print('Epoch finished ! Loss: {}'.format(loss_epoch))

        loss_list.append(loss_epoch)
        if loss_min is None:
            loss_min = loss_epoch
        if loss_epoch < loss_min:
            print('Update best epoch to {}'.format(epoch))
            loss_min = loss_epoch
            best_epoch = epoch
            best_model = net.state_dict()
    finish_time = time.time()
    print('Training finished in {}mins!'.format(round((finish_time-start_time)/60, 2)))

    final_model = net.state_dict()
    return loss_list, best_epoch, best_model, final_model
    