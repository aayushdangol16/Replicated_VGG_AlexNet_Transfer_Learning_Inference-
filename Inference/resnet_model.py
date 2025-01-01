import torch
import torch.nn as nn
from torchvision.models import resnet50,ResNet50_Weights





model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc=nn.Sequential(nn.Linear(in_features=2048,out_features=1),nn.Sigmoid())
state_dict = torch.load('resnet.pth',map_location=torch.device('cpu'),weights_only=True)
model.load_state_dict(state_dict)

for params in model.parameters():
  params.requires_grad_(False)

