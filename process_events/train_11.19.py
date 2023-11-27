# Author: Chun Tao
# Date: 11.19.2023

#%% Imports
import argparse
from dataloader_events import RoadEventDataset
import json
from models import ModifiedResNet18
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from utils import split_train_val_test, calculate_normalization, visualize_frame_bbox, initialize_weights
import pdb

#%% Arguments
parser = argparse.ArgumentParser()
# Data configs
parser.add_argument('-m', '--dataset-metafile-path', type=str, default='', required=True, help='path to the metafile json file')
parser.add_argument('-d', '--data-folder', type=str, default='', required=True, help='path to data folder containing image and motion files')
parser.add_argument('--new-splits', type=int, default=0, required=False, help='True (1) to split with new seed and save, False (0) to load splits dict')
# Train configs
parser.add_argument('-s', '--seed', type=int, default=0, required=False, help='the SEED number for stochasticity')
parser.add_argument('-e', '--num-epochs', type=int, default=10, required=True, help='number of epochs to train the model')

args = parser.parse_args()

#%% Seeds & Device
SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benmarks=False
os.environ['PYTHONHASHSEED']=str(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##% Perform train, val, test split (70-15-15)
if args.new_splits:
    print("Splitting dataset according to input SEED......")
    split_dict = split_train_val_test(metafile_path=args.dataset_metafile_path, 
                                      data_folder_path=args.data_folder, 
                                      seed=SEED,
                                      split_ratio=[0.7, 0.15, 0.15]) # total size (3005,); train size (2103,)
    print("Saving the new splits......")
    with open(os.path.join(args.data_folder, 'datafiles', args.dataset_metafile_path.split('/')[-1].split('.')[0]+'_splits.json'), 'w') as f:
        json.dump(split_dict, f)
else:
    print("Loading train-val-test splits from .json file......")
    with open(os.path.join(args.data_folder, 'datafiles', args.dataset_metafile_path.split('/')[-1].split('.')[0]+'_splits.json'), 'r') as f:
        split_dict = json.load(f)

#%% Find RGB mean and std for images from the training set
# print("Calculating RGB mean and std from the training set......")
# mean, std = calculate_normalization(metafile_path=args.dataset_metafile_path, 
#                                     data_folder_path=args.data_folder,
#                                     split_dict=split_dict)

#%% RoadEvent dataset and dataloader
# Train and test configs
dataset_conf_train = {'frame_type': "undistorted_rv",
                      'frame_size': (256, 256),
                      'batch_size': 64,
                      'shuffle': True,
                      'num_workers': 8,
                      'normalization': {'mean':[0.4180, 0.4343, 0.4137],
                                        'std': [0.2419, 0.2440, 0.2479]}} # find RGB min and std over the dataset images
dataset_conf_test = {'frame_type': "undistorted_rv",
                     'frame_size': (256, 256),
                     'batch_size': 128,
                     'shuffle': False,
                     'num_workers': 8,
                     'normalization': {'mean':[0.4180, 0.4343, 0.4137], # same as train set stats
                                       'std': [0.2419, 0.2440, 0.2479]}}

train_dataset = RoadEventDataset(dataset_metafile_path=args.dataset_metafile_path,
                                 data_folder_path=args.data_folder,
                                 split_dict=split_dict,
                                 dataset_conf=dataset_conf_train,
                                 split='train') 
val_dataset = RoadEventDataset(dataset_metafile_path=args.dataset_metafile_path,
                               data_folder_path=args.data_folder,
                               split_dict=split_dict,
                               dataset_conf=dataset_conf_test,
                               split='val')
test_dataset = RoadEventDataset(dataset_metafile_path=args.dataset_metafile_path,
                                data_folder_path=args.data_folder,
                                split_dict=split_dict,
                                dataset_conf=dataset_conf_test,
                                split='test')

print(f"\nCreating datasets...... Event count: {len(train_dataset.event_ids)}")
print(f'\tTrain set created...... Frame/sample count: {train_dataset.__len__()}')
print(f'\tVal set created...... Frame/sample count: {val_dataset.__len__()}')
print(f'\tTest set created...... Frame/sample count: {test_dataset.__len__()}')
frame, labels = train_dataset.__getitem__(0)
num_classes = sum([labels[i].shape[0] if i != 1 else 1 for i in range(len(labels))])
print(f'Frame size: {frame.shape}')
print(f'Labels: rotated rect bbox: {labels[0].shape}, event cls label: {labels[1].shape}, wheelAccel characteristics labels: {labels[2].shape}')
# 11.26: visualize frame and labels in one saved image
# visualize_frame_bbox(frame_tensor=frame,
#                      mean=dataset_conf_train['normalization']['mean'], 
#                      std=dataset_conf_train['normalization']['std'],
#                      bbox_tensor=labels[0],
#                      save_name=f"{train_dataset.frame_list[0]}")

# 11.27: visualize more frames and labels
# for frame_idx in range(20):
#     frame, labels, frame_name = train_dataset.__getitem__(frame_idx)
#     visualize_frame_bbox(frame_tensor=frame,
#                          mean=dataset_conf_train['normalization']['mean'], 
#                          std=dataset_conf_train['normalization']['std'],
#                          bbox_tensor=labels[0],
#                          save_name=f"{train_dataset.frame_list[frame_idx]}")

# Initiate RoadEvent DataLoader
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=dataset_conf_train['batch_size'],
                              shuffle=dataset_conf_train['shuffle'],
                              num_workers=dataset_conf_train['num_workers'])
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=dataset_conf_test['batch_size'],
                             shuffle=dataset_conf_test['shuffle'],
                             num_workers=dataset_conf_test['num_workers'])
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=dataset_conf_test['batch_size'],
                             shuffle=dataset_conf_test['shuffle'],
                             num_workers=dataset_conf_test['num_workers'])

#%% Import model and initialize weights
pretrained_dict = torchvision.models.resnet18(pretrained=True).state_dict()
model = ModifiedResNet18(num_classes=num_classes)
model_dict = model.features.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.features.load_state_dict(model_dict)
model.fc.apply(initialize_weights)
model = model.to(device)
print(model)

#%% Optimizer and loss criteria (11.27)
criterion_reg = nn.MSELoss() # switch to IOU loss later
criterion_cls = nn.BCEWithLogitsLoss(reduction='mean')
# loss_ratio = [0.25, 0.25, 0.25, 0.25]

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay, amsgrad=False)

scheduler = ExponentialLR(optimizer, gamma=0.9) # pick gamma less than 1

exp_name = f"EventResnet_{args.num_epochs}_{args.learning_rate}_{args.train_mode}"
exp_path = os.path.join(args.model_save_path, exp_name)
if not os.path.exists(exp_path): os.makedirs(exp_path)
# if args.resume_from_checkpoint==0:

#%% Train loop
model.train()
print("\nStart training!")
for epoch_idx, epoch in enumerate(range(args.num_epochs)):
    print(f"Epoch {epoch_idx+1:d}/{args.num_epochs:d}")
    epoch_loss_total = []
    epoch_loss_mse_center = []
    epoch_loss_mse_size = []
    epoch_loss_yaw = []
    epoch_loss_iou = []
    epoch_loss_cls = []
    epoch_loss_mse_wheelAccel = []

    for batch_idx, (images, labels) in enumerate(train_dataloader):
        # batch input and batch labels
        images, bboxes, cls_lbls, motion_lbls = images.to(device), labels[0].to(device), labels[1].to(device), labels[2].to(device)
        pdb.set_trace()

        # feedforward
        optimizer.zero_grad()
        out_reg, out_cls = model([spec, img])
        out_reg = torch.clamp(out_reg, 0, 255)
        loss_reg = criterion_reg(out_reg, bbox)
        loss_cls = criterion_cls(out_cls, event_label)
        loss = loss_reg + loss_cls
    
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # accumulate iteration loss
        if args.train_mode == "reg_and_cls" or args.train_mode == "only_reg":
            epoch_loss_reg_train += loss_reg.item()
        if args.train_mode == "reg_and_cls" or args.train_mode == "only_cls":
            epoch_loss_cls_train += loss_cls.item()
            _, predicted = torch.max(out_cls, 1)
            num_correct += sum(predicted==event_label).item()
        train_loss_tally.append(loss.item())

        # iteration printouts
        if (batch_idx+1) % 10 == 0:
            print(f"\t{batch_idx+1}/{len(train_dataloader)}")
        if args.train_mode == "reg_and_cls":
            if (batch_idx+1) == 20:
                plot_image(img[:4], bbox[:4], event_label[:4], out_reg[:4], out_cls[:4], eventType_dict, os.path.join(exp_path, f"epoch_{epoch_idx+1}.png"))
   
        scheduler.step()
        acc_cls = num_correct / train_size
        train_acc_tally.append(acc_cls)
        print(f"\tEpoch {epoch_idx+1:d}/{args.num_epochs:d} done, LR {scheduler.get_last_lr()[0]:.2e} \
                \n\tAvg train regression loss: {epoch_loss_reg_train/len(train_dataloader):.3f}, Avg train classification loss: {epoch_loss_cls_train/len(train_dataloader):.3f}")
        print(f"\tTrain classification accuracy: {num_correct:d}/{train_size:d}, {(acc_cls*100):.3f}%")
        
        pdb.set_trace()
