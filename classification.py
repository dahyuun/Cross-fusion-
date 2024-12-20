import os
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

import cv2
import torch
import glob
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import timm
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Custom transformation for center cropping images
class CenterCropSquare(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        width, height = image.size
        new_edge_length = min(width, height)
        left = (width - new_edge_length) / 2
        top = (height - new_edge_length) / 2
        right = (width + new_edge_length) / 2
        bottom = (height + new_edge_length) / 2
        image = image.crop((left, top, right, bottom))
        image = T.Resize(image, (self.output_size, self.output_size))
        return image

# Dataset class for Diabetic Retinopathy classification
class DRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        for label, class_dir in enumerate(["0", "1"]):
            for img_path in glob.glob(os.path.join(root_dir, class_dir, "*.jpeg")):
                self.image_paths.append(img_path)
                self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Segmentation model class definition
class SwinUNet(nn.Module):
    def __init__(self):
        super(SwinUNet, self).__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True, out_indices=(0, 1, 2, 3), img_size=640)

        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.final_up = nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3 = self.swin(x)

        x3 = x3.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x1 = x1.permute(0, 3, 1, 2)
        x0 = x0.permute(0, 3, 1, 2)

        x = self.up1(x3)
        x = self.conv1(x + x2)
        x = self.up2(x)
        x = self.conv2(x + x1)
        x = self.up3(x)
        x = self.conv3(x + x0)
        x = self.up4(x)
        x = self.conv4(x)
        x = self.final_up(x)
        x = self.final_conv(x)
        return x

# Cross attention block definition
class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super(CrossAttentionBlock, self).__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        return out

# Classification model with cross-attention block
class CustomSwinTransformer(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, dropout_rate=0.45):
        super(CustomSwinTransformer, self).__init__()
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=3, img_size=640)
        self.cross_attention = CrossAttentionBlock()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self, x, lesion_map):
        x = self.base_model(x)
        lesion_map = torch.cat([lesion_map, lesion_map, lesion_map], dim=1)  # Convert 1 channel to 3 channels
        lesion_map = self.base_model(lesion_map)
        x = self.cross_attention(x, lesion_map, lesion_map)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize and load models
segmentation_model = SwinUNet().cuda()
classification_model_name = 'swin_tiny_patch4_window7_224'
classification_model = CustomSwinTransformer(classification_model_name, num_classes=1, pretrained=True, dropout_rate=0.45).cuda()

# Load pretrained weights for the segmentation model
segmentation_weights_path = './model/segmentation.ckpt'
segmentation_model.load_state_dict(torch.load(segmentation_weights_path))

# Training settings
data_dir = './data/original/EyePACS/'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
num_epochs = 50
criterion = nn.BCEWithLogitsLoss()  # Updated loss function
base_lr = 1e-3

# Data preparation
valid_transform = T.Compose([T.Resize((640, 640)), T.ToTensor(), T.Normalize(mean, std)])
train_transform = T.Compose([T.Resize((640, 640)), T.RandomHorizontalFlip(), T.RandomRotation(15), T.ToTensor(), T.Normalize(mean, std)])

val_dataset = DRDataset(data_dir + 'validation', valid_transform)
train_dataset = DRDataset(data_dir + 'train', train_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=False)  # Reduced batch size
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)  # Reduced batch size

optimizer = SGD(params=classification_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
steps_per_epoch = len(train_loader)
max_iterations = num_epochs * steps_per_epoch

scaler = GradScaler()  # Mixed precision training

for epoch in range(num_epochs):
    classification_model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets = torch.unsqueeze(targets, 1).float()

        # Generate lesion maps using the segmentation model
        with torch.no_grad():
            lesion_maps = segmentation_model(inputs)

        optimizer.zero_grad()
        with autocast():  # Mixed precision training
            outputs = classification_model(inputs, lesion_maps)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr = base_lr * (1.0 - float(epoch * len(train_loader) + i) / max_iterations)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], LR: {lr:.5f}, Loss: {loss.item():.4f}')

    # Validation phase
    classification_model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            lesion_maps = segmentation_model(inputs)
            outputs = classification_model(inputs, lesion_maps)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

torch.save(classification_model.state_dict(), './model/final_classification.pth')
print('Final model saved: final_classification.pth')