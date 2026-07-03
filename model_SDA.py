import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNetWithSD(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Load Pre-trained ResNet18
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Separate the feature extraction backbone
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Auxiliary classifier for each layer (Linear layer)
        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(256, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        def get_logits(feat, fc):
            # Apply Global Average Pooling (GAP) followed by a fully connected (FC) layer
            out = F.adaptive_avg_pool2d(feat, (1, 1))
            return fc(torch.flatten(out, 1))

        logits = [
            get_logits(l1, self.fc1), 
            get_logits(l2, self.fc2), 
            get_logits(l3, self.fc3), 
            get_logits(l4, self.fc4)
        ]
        feats = [l1, l2, l3, l4]
        return logits, feats

class MultiSDAdapter(nn.Module):
    def __init__(self, channels=[64, 128, 256], teacher_dim=512):
        super().__init__()        
        # Align the channel dimensions of the student layers
        # to match the teacher (L4) feature dimension (512)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Flatten(), 
                nn.Linear(c, teacher_dim), 
                nn.ReLU()
            ) for c in channels
        ])
    def forward(self, i, x): 
        return self.adapters[i](x)
