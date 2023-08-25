import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.nn.init as init
import torchvision

class EventNN(nn.Module):
    def __init__(self):
        super(EventNN, self).__init__()
        self.conv_seqn_v = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool_v = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_seqn_s = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fc_seqn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(64, 13),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3)
        )


    def forward(self, x): # x[0] is spec (1, 17, 31), x[1] is frame (3, 256, 256)
        feat_v = self.conv_seqn_v(x[1]) 
        feat_v = self.global_avg_pool_v(feat_v) 
        feat_v = feat_v.view(-1, 64)

        feat_s = self.conv_seqn_s(x[0])
        feat_s = feat_s.view(-1, 64)

        # concat
        feat_a = torch.cat((feat_s, feat_v), dim=-1)
        out = self.fc_seqn(feat_a)

        return out[:, :8], out[:, -5:]
    

class EventResnet(nn.Module):
    def __init__(self):
        super(EventNN, self).__init__()
        self.conv_seqn_v = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool_v = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_seqn_s = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fc_seqn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(64, 13),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3)
        )


    def forward(self, x): # x[0] is spec (1, 17, 31), x[1] is frame (3, 256, 256)
        feat_v = self.conv_seqn_v(x[1]) 
        feat_v = self.global_avg_pool_v(feat_v) 
        feat_v = feat_v.view(-1, 64)

        feat_s = self.conv_seqn_s(x[0])
        feat_s = feat_s.view(-1, 64)

        # concat
        feat_a = torch.cat((feat_s, feat_v), dim=-1)
        out = self.fc_seqn(feat_a)

        return out[:, :8], out[:, -5:]
    

# custom ResNet-18 model
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=13):
        super(ModifiedResNet18, self).__init__()
        # Load the pre-trained ResNet-18 model (excluding the last layer)
        pretrained_resnet = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(pretrained_resnet.children())[:-1])
        # Add a new linear layer for the modified number of classes
        self.fc = nn.Linear(512, num_classes)  # 512 is the output size of the ResNet-18 before the last layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x