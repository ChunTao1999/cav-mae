# Author: Chun Tao
# Date: 11.19.2023

#%% Imports
import argparse
from dataloader_events import RoadEventDataset
import json
from loss_iou_giou import cal_giou, cal_diou, plot_corners
import numpy as np
import os
import random
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.tensorboard import SummaryWriter
from utils import transform_bbox, split_train_val_test, calculate_normalization, visualize_frame_bbox, initialize_weights
import pdb # for debug


#%% Arguments
parser = argparse.ArgumentParser()
# Data configs
parser.add_argument('-m', '--dataset-metafile-path', type=str, default='', required=True, help='path to the metafile json file')
parser.add_argument('-d', '--data-folder', type=str, default='', required=True, help='path to data folder containing image and motion files')
parser.add_argument('--new-splits', type=int, default=0, required=False, help='True (1) to split with new seed and save, False (0) to load prev splits dict')
# Train configs
parser.add_argument('-s', '--seed', type=int, default=0, required=False, help='the SEED number for stochasticity')
parser.add_argument('-e', '--num-epochs', type=int, default=10, required=False, help='number of epochs to train the model')
parser.add_argument('--start-lr', type=float, default=1e-3, required=False, help='start learning rate')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.99), required=False, help='adam betas')
parser.add_argument('--eps', type=float, default=1e-8, required=False, help='adam eps')
parser.add_argument('--weight-decay', type=float, default=1e-3, required=False, help='adam weight decay')
parser.add_argument('--exp-scheduler-gamma', type=float, default=0.9, required=False, help='gamma value for exponential scheduler')

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
                     'batch_size': 64,
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
# test_dataset = RoadEventDataset(dataset_metafile_path=args.dataset_metafile_path,
#                                 data_folder_path=args.data_folder,
#                                 split_dict=split_dict,
#                                 dataset_conf=dataset_conf_test,
#                                 split='test')

print(f"\nCreating datasets...... Event count: {len(train_dataset.event_ids)}")
print(f'\tTrain set created...... Frame/sample count: {train_dataset.__len__()}')
# print(f'\tVal set created...... Frame/sample count: {val_dataset.__len__()}')
# print(f'\tTest set created...... Frame/sample count: {test_dataset.__len__()}')
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
# test_dataloader = DataLoader(dataset=test_dataset,
#                              batch_size=dataset_conf_test['batch_size'],
#                              shuffle=dataset_conf_test['shuffle'],
#                              num_workers=dataset_conf_test['num_workers'])

#%% Import model and initialize weights
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                trainable_backbone_layers=3) # default
model.to(device)
print(model)
# summary(model, input_size=(64, 3, dataset_conf_train['frame_size'][0], dataset_conf_train['frame_size'][1]))
# pdb.set_trace()

#%% Optimizer and loss criteria (11.27)
criterion_mse = nn.MSELoss(reduction='mean') 
# can deal with data imbalance (11.29)
# pos_weight = torch.tensor(train_dataset.loss_ratio, dtype=torch.float)
pos_weight = torch.tensor([train_dataset.neg_count / train_dataset.pos_count], dtype=torch.float).to(device)
criterion_bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
# loss ratio for different losses
loss_ratio = [1, 2, 1] # (box, cls, motion)
# loss ratio for bboxes
loss_bbox_ratio = [0.1, 0.5, 1.5, 2] # (box_center, box_size, box_yaw, box_iou)

optimizer = optim.Adam(model.parameters(), lr=args.start_lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay, amsgrad=False)
scheduler = ExponentialLR(optimizer, gamma=args.exp_scheduler_gamma)

exp_name = f"Fasterrcnn_{args.start_lr}_{args.num_epochs}_{args.exp_scheduler_gamma}_{loss_ratio[0]}_{loss_ratio[1]}_{loss_ratio[2]}_{loss_bbox_ratio[0]}_{loss_bbox_ratio[1]}_{loss_bbox_ratio[2]}_{loss_bbox_ratio[3]}"
exp_path = os.path.join("trained_models", exp_name)
if not os.path.exists(exp_path): os.makedirs(exp_path)
# if args.resume_from_checkpoint==0:

#%% Train loop
# Tensorboard
writer = SummaryWriter(f"logs/{exp_name}")
# writer.add_graph(model, torch.randn(64, 3, 256, 256).to(device))
# pdb.set_trace()

model.train()
print("\nStart training!")
train_loss_tally = []
train_acc_tally = []

for epoch_idx, epoch in enumerate(range(args.num_epochs)):
    print(f"Epoch {epoch_idx+1:d}/{args.num_epochs:d}")
    epoch_loss_total = []
    epoch_loss_box_center = []
    epoch_loss_box_size = []
    epoch_loss_box_yaw = []
    epoch_loss_box_iou = []
    epoch_loss_cls = []
    epoch_loss_motion = []
    num_correct = 0

    model.train()
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        ite_num = epoch*len(train_dataloader)+batch_idx

        # batch input and batch labels
        images, bboxes, cls_lbls, motion_lbls = images.to(device), labels[0].to(device), labels[1].to(device), labels[2].to(device)
        images = list(image for image in images)

        # For fasterrcnn
        # transform (x_c, y_c, w, h, alpha) bboxes to (x1, y1, x2, y2) format
        pdb.set_trace()
        bboxes = transform_bbox(bboxes)
        pdb.set_trace()


        # feedforward
        outputs = model(images)
        out_bboxes, out_cls, out_motion = outputs[:, :5], outputs[:, 5], outputs[:, 6:]
        pdb.set_trace()

        # loss_bbox is composed of 3 parts: 1) mse for box size and box center coords 2) yaw loss for angle 3) diou loss
        loss_box_center = loss_bbox_ratio[0] * criterion_mse(out_bboxes[:,0:2], bboxes[:,0:2]) 
        loss_box_size = loss_bbox_ratio[1] * criterion_mse(out_bboxes[:,3:], bboxes[:,3:])# center and size mse
        loss_box_yaw = loss_bbox_ratio[2] * ((torch.abs(torch.sin(out_bboxes[:,2] - bboxes[:,2]))).mean())
        out_bboxes, bboxes = torch.unsqueeze(out_bboxes[:,[0,1,3,4,2]], 1), torch.unsqueeze(bboxes[:,[0,1,3,4,2]], 1) # permute to the (x_c, y_c, w, h, alpha) order
        loss_box_iou, ious = cal_diou(out_bboxes, bboxes)
        loss_box_iou = loss_bbox_ratio[3] * (torch.mean(loss_box_iou))
        loss_box = loss_box_center + loss_box_size + loss_box_yaw + loss_box_iou

        # loss_cls & loss_motion
        loss_cls = criterion_bce(out_cls, cls_lbls)
        loss_motion = criterion_mse(out_motion, motion_lbls)

        # total loss wtih weighted losses added
        loss = loss_ratio[0] * loss_box + loss_ratio[1] * loss_cls + loss_ratio[2] * loss_motion
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = torch.clamp(param.grad.data, -1.0, 1.0)
        optimizer.step()

        # iteration logging
        writer.add_scalar('Loss/train_total', loss.item(), ite_num)
        writer.add_scalar('Loss/train_box_center', loss_box_center.item(), ite_num)
        writer.add_scalar('Loss/train_box_size', loss_box_size.item(), ite_num)
        writer.add_scalar('Loss/train_box_yaw', loss_box_yaw.item(), ite_num)
        writer.add_scalar('Loss/train_box_iou', loss_box_iou.item(), ite_num)
        writer.add_scalar('Loss/train_cls', loss_cls.item(), ite_num)
        writer.add_scalar('Loss/train_motion', loss_motion.item(), ite_num)
        num_correct_batch = sum((torch.sigmoid(out_cls)>=0.5).float()==cls_lbls)
        writer.add_scalar('Accuracy/train_cls', num_correct_batch.item()/images.shape[0]*100, ite_num)
        # add iou>0.5 percentage 
        ratio_iou_batch = sum(ious>=0.5)/images.shape[0]*100
        writer.add_scalar('Accuracy/train_iou>=0.5', ratio_iou_batch.item(), ite_num)
        # pdb.set_trace()

        # accumulate iteration loss to epoch loss
        epoch_loss_total.append(loss.detach().cpu().numpy())
        epoch_loss_box_center.append(loss_box_center.detach().cpu().numpy())
        epoch_loss_box_size.append(loss_box_size.detach().cpu().numpy())
        epoch_loss_box_yaw.append(loss_box_yaw.detach().cpu().numpy())
        epoch_loss_box_iou.append(loss_box_iou.detach().cpu().numpy())
        epoch_loss_cls.append(loss_cls.detach().cpu().numpy())
        epoch_loss_motion.append(loss_motion.detach().cpu().numpy())
        
        # for accuracy
        num_correct += num_correct_batch.item()

        # iteration printouts
        if (batch_idx+1) % 10 == 0:
            print(f"\t{batch_idx+1}/{len(train_dataloader)}")
        
    # End of epoch
    scheduler.step()
    acc_cls = num_correct / len(train_dataset)
    train_acc_tally.append(acc_cls)
    train_loss_tally.append(np.mean(epoch_loss_total))
    print(f"\tEpoch {epoch_idx+1:d}/{args.num_epochs:d} done, LR {scheduler.get_last_lr()[0]:.2e} \
            \n\tLoss: {np.mean(epoch_loss_total):.3f}, Accuracy: {(acc_cls*100):.3f}%")
        
    # save model every 10 epochs
    if (epoch_idx+1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(exp_path, f"epoch_{epoch_idx+1}.pth"))
        print(f"Model saved......, Epoch {epoch_idx+1}/{args.num_epochs}")
    print("Finished training!")


#%% Validation loop
model.eval()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_dataloader):
        ite_num = epoch*len(train_dataloader)+batch_idx
        # move images and labels to device
        images, bboxes, cls_lbls, motion_lbls = images.to(device), labels[0].to(device), labels[1].to(device), labels[2].to(device)

        # feedforward
        outputs = model(images)
        out_bboxes, out_cls, out_motion = outputs[:, :5], outputs[:, 5], outputs[:, 6:]
        
        # loss_bbox is composed of 3 parts: 1) mse for box size and box center coords 2) yaw loss for angle 3) diou loss
        loss_box_center = loss_bbox_ratio[0] * criterion_mse(out_bboxes[:,0:2], bboxes[:,0:2]) 
        loss_box_size = loss_bbox_ratio[1] * criterion_mse(out_bboxes[:,3:], bboxes[:,3:])# center and size mse
        loss_box_yaw = loss_bbox_ratio[2] * ((torch.abs(torch.sin(out_bboxes[:,2] - bboxes[:,2]))).mean())
        out_bboxes, bboxes = torch.unsqueeze(out_bboxes[:,[0,1,3,4,2]], 1), torch.unsqueeze(bboxes[:,[0,1,3,4,2]], 1) # permute to the (x_c, y_c, w, h, alpha) order
        loss_box_iou, ious = cal_diou(out_bboxes, bboxes)
        loss_box_iou = loss_bbox_ratio[3] * (torch.mean(loss_box_iou))
        loss_box = loss_box_center + loss_box_size + loss_box_yaw + loss_box_iou

        # loss_cls & loss_motion
        loss_cls = criterion_bce(out_cls, cls_lbls)
        loss_motion = criterion_mse(out_motion, motion_lbls)

        # total loss wtih weighted losses added
        writer.add_scalar('Loss/val_total', loss.item(), ite_num)
        writer.add_scalar('Loss/val_box_center', loss_box_center.item(), ite_num)
        writer.add_scalar('Loss/val_box_size', loss_box_size.item(), ite_num)
        writer.add_scalar('Loss/val_box_yaw', loss_box_yaw.item(), ite_num)
        writer.add_scalar('Loss/val_box_iou', loss_box_iou.item(), ite_num)
        writer.add_scalar('Loss/val_cls', loss_cls.item(), ite_num)
        writer.add_scalar('Loss/val_motion', loss_motion.item(), ite_num)
        num_correct_batch = sum((torch.sigmoid(out_cls)>=0.5).float()==cls_lbls)
        writer.add_scalar('Accuracy/val_cls', num_correct_batch.item()/images.shape[0]*100, ite_num)
        ratio_iou_batch = sum(ious>=0.5)/images.shape[0]*100
        writer.add_scalar('Accuracy/val_iou>=0.5', ratio_iou_batch.item(), ite_num)
        
    print("Finished validation!")
        
#%% Test loop
writer.close()



images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (4, 11))
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

output = model(images, targets)
pdb.set_trace()