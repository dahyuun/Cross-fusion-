import os
import cv2
import argparse
import time
import tqdm
import numpy as np
from curses import raw
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import glob
import timm

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

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

# Initialize models
segmentation_model = SwinUNet().cuda()
classification_model_name = 'swin_tiny_patch4_window7_224'
classification_model = CustomSwinTransformer(classification_model_name, num_classes=1, pretrained=True, dropout_rate=0.45).cuda()

# Load pretrained weights
segmentation_weights_path = './model/seg_0521_5.ckpt'
classification_weights_path = './model/final_trans_0729_2.pth'
segmentation_model.load_state_dict(torch.load(segmentation_weights_path))
classification_model.load_state_dict(torch.load(classification_weights_path))

print(f'{classification_model_name} tested !!!')
data_dir = './data/original/EyePACS/'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Data transformation and loading
test_transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean, std)
])

test_dataset = DRDataset(data_dir + 'test2', test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=False)

classification_model = classification_model.cuda()
classification_model.eval()

sigmoid = nn.Sigmoid()

preds = []
all_targets = []
total_time = 0.0

# Testing loop
for inputs, targets in tqdm.tqdm(test_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    time1 = time.time()

    with torch.no_grad():
        lesion_maps = segmentation_model(inputs)  # Generate lesion maps
        outputs = sigmoid(classification_model(inputs, lesion_maps))  # Use cross-attention classification model

    total_time += (time.time() - time1)

    preds.append(outputs.cpu().detach())
    all_targets.append(targets.cpu().detach())

preds = torch.cat(preds).cpu().numpy()
all_targets = torch.cat(all_targets).cpu().numpy()
final_preds = preds > 0.5
acc = accuracy_score(all_targets, final_preds)
print('Accuracy: ', acc)

# ROC Curve and AUC computation and visualization
fpr, tpr, thresholds = roc_curve(all_targets, preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./modeltest_figure/0729_2_roc_curve.png')  # Save ROC curve to file
plt.show()

# Confusion Matrix computation and visualization
cm = confusion_matrix(all_targets, final_preds)
print('Confusion Matrix:\n', cm)  # Print confusion matrix values
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['NonReferable DR', 'ReferableDR'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('./modeltest_figure/0729_2_confusion_matrix.png')  # Save confusion matrix to file
plt.show()

# Performance calculation
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

# Print evaluation metrics
print('AUC:', roc_auc)
print('TPR (Sensitivity):', TPR)
print('TNR (Specificity):', TNR)
