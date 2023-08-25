# Author: Chun Tao

#%% Imports
import argparse
import json
import os
import random
from PIL import Image
import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from dataloader_events import RoadEventDataset
from preprocess_data import preprocess
from utils import calibrate_camera, define_perspective_transform, plot_image, plot_loss_curves
from models import EventNN, ModifiedResNet18
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pdb # debug

#%% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--calibrate', type=int, default=0, required=True, help='whether to calibrate camera and save camera matrices, 0 for True')
parser.add_argument('-p', '--perspective', type=int, default=0, required=True, help='whether to compute perspective transform or not, 0 for True')
parser.add_argument('-j', '--dataset-jsonfile-path', type=str, default='', required=True, help='filepath to dataset jsonfile')
parser.add_argument('-s', '--seed', type=int, default=0, required=True, help='the seed number to use for the current runtime')
parser.add_argument('-e', '--num-epochs', type=int, default=1, required=True, help='number of epochs for training')
parser.add_argument('--preprocess', type=int, default=0, required=True, help='whether or not to preprocess wheelAccels and event frames, 0 for True')
parser.add_argument('--eventtype-json-path', type=str, default='', required=True, help='path to the json file describing convertion from event type label to event type description')
parser.add_argument('-d', '--data-path', type=str, default='', required=True, help='path to the data folder')
parser.add_argument('--cal-data-path', type=str, default='', required=True, help='path to the saved calibration data')
parser.add_argument('--dataset-path', type=str, default='', required=True, help='dataset path to save to')
parser.add_argument('--download-csvs', type=int, default=0, required=True, help='whether to download session csvs for sensor data, or to use a past stored session csv, 0 for True')
parser.add_argument('--session-id', type=int, default=75151, required=False, help='session ids to the session data')
parser.add_argument('--wheelaccel-timespan', type=float, default=1.0, required=True, help='the timespan of WheelAccel segments for each event')
# training configs
parser.add_argument('--resume-from-checkpoint', type=int, default=1, required=False, help='whether to start from saved checkpoint or not, 0 for True')
parser.add_argument('--saved-checkpoint-path', type=str, default='', required=False, help='path to the saved checkpoint, if want to resume training')
parser.add_argument('--learning-rate', type=float, default=1e-3, required=False, help='train learning rate')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), required=False, help='adam betas')
parser.add_argument('--eps', type=float, default=1e-8, required=False, help='adam eps')
parser.add_argument('--weight-decay', type=float, default=1e-3, required=False, help='adam weight decay')
parser.add_argument('--model-save-path', type=str, default='', required=True, help='path to save the trained model parameters')
args = parser.parse_args()


#%% Seeds
SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benmarks=False
os.environ['PYTHONHASHSEED']=str(SEED)
#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Preprocesses
# if needed, calibrate the camera and save camera matrices
if not os.path.exists(os.path.join(args.data_path, "results")):
    os.makedirs(os.path.join(args.data_path, "results"))
if args.calibrate==0:
    print("Preprocess: camera calibration......")
    calibrate_camera(objp_w=10,
                     objp_h=7,
                     cal_data_path=args.cal_data_path,
                     save_path=os.path.join(args.data_path, "results"))
else:
    print("Preprocess: camera calibration skipped, using previous......")
# define the perspective transform to use for the raw rv road event frames
if args.perspective==0:
    print("\nPreprocess: perspective transform......")
    define_perspective_transform(cal_data_path=args.cal_data_path,
                                 img_cal_path=os.path.join(args.data_path, 'cal_img.jpg'),
                                 IMAGE_H=600, 
                                 IMAGE_W=600,
                                 cv2_imread_frame_dim=(1920, 1080),
                                 srcpts_arr = [[236,1218],[2170,1190],[1532,703],[885,708]], # CCLK, from bottom left
                                 destpts_arr= [[50,580],[550,580],[550,250],[50,250]])
else:
    print("\nPreprocess: perspective transform skipped, using previous......")
# define wheelAccel transform configs
wheelAccel_conf = {'wheel_id': ['rlWheelAccel', 'rrWheelAccel'],
                   'sampling_freq': 500,
                   'timespan': args.wheelaccel_timespan,
                   'normalize': True,
                   'nominal_speed': 5,
                   'N_windows_fft': 32,
                   'noverlap': 16,
                   'spec_size': (17, 31)}
eventMarking_conf = {'cv2_imread_frame_dim': (1920, 1080), # (w, h)
                     'event_timestamp_shift': 0, # negative shift here
                     'frame_timestamp_shift': 0.2,
                     'bev_frame_dim': (600,600),
                     'wheel_to_base_dist': 3.5, # 4.572
                     'base_pixel': 20,
                     'wheel_width': 1.664,
                     'xm_per_pix': 4.318/500,
                     'ym_per_pix': 8.8/330, # 8.89
                     'event_len_pix': 200}

# Preprocess to get event frames, event 1-d wheelAccel, event 2-d spectrograms, and grouth-truth bbox locations and event labels
# Write all info to the dataset dictionary
if args.preprocess==0:
    print("\nPreprocess: transforming wheelAccel to spectrogram and marking events in the frames......")
    preprocess(cal_data_path=args.cal_data_path,
               data_path=args.data_path,
               save_path=args.dataset_path,
               json_save_path = args.dataset_jsonfile_path,
               wheelAccel_conf=wheelAccel_conf,
               eventmarking_conf=eventMarking_conf,
               eventType_json_path=args.eventtype_json_path,
               download=True if args.download_csvs==0 else False,
               plot_veh_speed_yawrate=False,
               plot_wheelAccel=True,
               plot_processedFrames=True)
else:
    print("\nPreprocess: transforming wheelAccel to spectrogram and marking events in the frames skipped, using existing dataset......")
# pdb.set_trace()


#%% RoadEvent Dataset
# Initiate RoadEvent Dataset
dataset_conf_train = {'frame_size': (256, 256),
                      'batch_size': 64,
                      'shuffle': True,
                      'num_workers': 8} # frame_size
dataset_conf_test = {'frame_size': (256, 256),
                      'batch_size': 64,
                      'shuffle': False,
                      'num_workers': 8} # frame_size
roadevent_dataset = RoadEventDataset(dataset_jsonfile_path=args.dataset_jsonfile_path,
                                     dataset_conf=dataset_conf_train) # pdb breakpoint inside
dataset_size = len(roadevent_dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(roadevent_dataset, [train_size, test_size])
print(f'\nDataset created...... Event count: {len(roadevent_dataset.event_ids)}; Frame/sample count: {roadevent_dataset.__len__()}')
print(f'\nTrain dataset created...... Frame/sample count: {train_dataset.__len__()}')
print(f'\nTest dataset created...... Frame/sample count: {test_dataset.__len__()}')
sample_spec, sample_frame, sample_label, sample_bbox = train_dataset.__getitem__(0)
print(f'\twheelAccel spec size: {sample_spec.shape}')
print(f'\tframe size: {sample_frame.shape}')

# Initiate RoadEvent DataLoader
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=dataset_conf_train['batch_size'],
                              shuffle=dataset_conf_train['shuffle'],
                              num_workers=dataset_conf_train['num_workers'])
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=dataset_conf_test['batch_size'],
                             shuffle=dataset_conf_test['shuffle'],
                             num_workers=dataset_conf_test['num_workers'])
# Event label dictionary
with open('./event_types_manual_label.json', 'r') as f:
    eventType_dict = json.load(f)


#%% Training and validation loop
# model init function
def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

# Create model
# model = EventNN()
# model.apply(initialize_weights)
# model = torchvision.models.resnet18(pretrained=True)
# num_classes = 5
# model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
pretrained_dict = torchvision.models.resnet18(pretrained=True).state_dict()
model = ModifiedResNet18(num_classes=8)
model_dict = model.features.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.features.load_state_dict(model_dict)
model.fc.apply(initialize_weights)
model = model.to(device)
print(model)
# pdb.set_trace()

criterion_reg = nn.MSELoss() # switch to IOU loss later
# class_weights = torch.tensor([0.2, 0.21, 0.23, 0.12, 0.24])
# class_weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
# class_weights = class_weights.to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay, amsgrad=False)
scheduler = ExponentialLR(optimizer, gamma=0.9)
print("\nStart Event Detection and Classification Training......\n")
exp_name = f"EventResnetreg_{args.num_epochs}_{args.learning_rate}_new"
exp_path = os.path.join(args.model_save_path, exp_name)
if not os.path.exists(exp_path): os.makedirs(exp_path)
# if args.resume_from_checkpoint==0:

# with torch.autograd.set_detect_anomaly(True):
train_loss_tally = []
test_loss_tally = []
train_acc_tally = []
test_acc_tally = []
for epoch_idx, epoch in enumerate(range(args.num_epochs)):
    print(f"Epoch {epoch_idx+1:d}/{args.num_epochs:d}")
    # Train loop
    model.train()
    epoch_loss_reg_train = 0.0
    epoch_loss_cls_train = 0.0
    num_correct = 0
    for batch_idx, data in enumerate(train_dataloader):
        # get input data
        spec, img = data[0].float().to(device), data[1].float().to(device) # spec (bs, 1, 17, 31); img (bs, 3, 256, 256)
        # get label data
        event_label, bbox = data[2].to(device), data[3].float().to(device)

        # feedforward
        optimizer.zero_grad()
        # out_reg, out_cls = model([spec, img])
        out_reg = model(img)
        # out_reg, out_cls = out_all[:, :8], out_all[:, -5:]

        loss_reg = criterion_reg(out_reg, bbox)
        # loss_cls = criterion_cls(out_cls, event_label)
        # loss_reg.backward(retain_graph=True)
        # loss_cls.backward()
        # loss = loss_reg + loss_cls
        loss = loss_reg
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss_reg_train += loss_reg.item()
        # epoch_loss_cls_train += loss_cls.item()
        train_loss_tally.append(loss.item())

        # _, predicted = torch.max(out_cls, 1)
        # num_correct += sum(predicted==event_label).item()
        if (batch_idx+1) % 10 == 0:
            print(f"\t{batch_idx+1}/{len(train_dataloader)}")
        if (batch_idx+1) == 20:
            plot_image(img[:4], bbox[:4], event_label[:4], out_reg[:4], event_label[:4], eventType_dict, os.path.join(exp_path, f"epoch_{epoch_idx+1}.png"))
    scheduler.step()
    acc_cls = num_correct / train_size
    train_acc_tally.append(acc_cls)
    # print(f"\tEpoch {epoch_idx+1:d}/{args.num_epochs:d} done, LR {scheduler.get_last_lr()[0]:.2e} \
    #         \n\tAvg train regression loss: {epoch_loss_reg_train/len(train_dataloader):.3f}, Avg train classification loss: {epoch_loss_cls_train/len(train_dataloader):.3f}")
    print(f"\tEpoch {epoch_idx+1:d}/{args.num_epochs:d} done, LR {scheduler.get_last_lr()[0]:.2e} \
            \n\tAvg train regression loss: {epoch_loss_reg_train/len(train_dataloader):.3f}")
    print(f"\tTrain classification accuracy: {num_correct:d}/{train_size:d}, {(acc_cls*100):.3f}%")

    # Test loop
    model.eval()
    epoch_loss_reg_test = 0.0
    epoch_loss_cls_test = 0.0
    num_correct = 0
    true_labels_list = []
    predicted_labels_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            spec, img = data[0].float().to(device), data[1].float().to(device)
            event_label, bbox = data[2].to(device), data[3].float().to(device)
            # out_reg, out_cls = model([spec, img])
            out_reg = model(img)
            # out_reg, out_cls = out_all[:, :8], out_all[:, -5:]

            loss_reg = criterion_reg(out_reg, bbox)
            # loss_cls = criterion_cls(out_cls, event_label)
            # loss = loss_reg + loss_cls
            loss = loss_reg
            epoch_loss_reg_test += loss_reg.item()
            # epoch_loss_cls_test += loss_cls.item()
            
            # _, predicted = torch.max(out_cls, 1)
            # num_correct += sum(predicted==event_label).item()
            # if epoch_idx==args.num_epochs-1:
            #     true_labels_list.append(event_label.clone().detach().cpu())
            #     predicted_labels_list.append(predicted.clone().detach().cpu())
    test_loss_tally.append(epoch_loss_reg_test/len(test_dataloader))
    acc_cls = num_correct / test_size
    test_acc_tally.append(acc_cls)
    # print(f"\tAvg test regression loss: {epoch_loss_reg_test/len(test_dataloader):.3f}, Avg test classification loss: {epoch_loss_cls_test/len(test_dataloader):.3f}")
    print(f"\tAvg test regression loss: {epoch_loss_reg_test/len(test_dataloader):.3f}")
    print(f"\tTest classification accuracy: {num_correct:d}/{test_size:d}, {(acc_cls*100):.3f}%")


# confusion matrix
# true_labels = torch.cat(true_labels_list, dim=0)
# predicted_labels = torch.cat(predicted_labels_list, dim=0)
# conf_matrix = confusion_matrix(true_labels.numpy(), predicted_labels.numpy(),)
# class_names = ['Pothole', 'Manhole Cover', 'Drain Gate', 'Unknown', 'Speed Bump']  # Replace with your actual class names
# class_dict = {i: class_name for i, class_name in enumerate(class_names)}
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=[class_dict[i] for i in range(len(class_names))],
#             yticklabels=[class_dict[i] for i in range(len(class_names))])
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.savefig(os.path.join(exp_path, "conf_mat.png"))
# plt.close()

# plot losses
np.save(os.path.join(exp_path, "train_loss.npy"), np.array(train_loss_tally))
np.save(os.path.join(exp_path, "test_loss.npy"), np.array(test_loss_tally))
np.save(os.path.join(exp_path, "train_acc.npy"), np.array(train_acc_tally))
np.save(os.path.join(exp_path, "test_acc.npy"), np.array(test_acc_tally))
plot_loss_curves(exp_path)

# save model
torch.save(model.state_dict(), os.path.join(exp_path, f"{args.num_epochs}_{args.learning_rate}.pth"))
print("Train and test finished!")


#%% TO-DOs:
# image color normalization
# IOU loss on bbox reconstruction and CE loss on event classification, produce MaP and Confusion Matrix
pdb.set_trace() 
