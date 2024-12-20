import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
import cv2
from PIL import ImageOps
from sklearn.metrics import jaccard_score as iou_score
from sklearn.metrics import f1_score
from torchvision.utils import save_image

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class ImageMaskTransform:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.image_transform = transforms.Compose([
            # transforms.Lambda(lambda img: Image.fromarray(self.apply_clahe(np.array(img)))),
            transforms.Lambda(lambda img: Image.fromarray(np.array(img))),  # CLAHE 적용 부분을 제거
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    # def apply_clahe(self, img):
    #     img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #     img_yuv[:,:,0] = self.clahe.apply(img_yuv[:,:,0])
    #     img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    #     return img_output

    def __call__(self, img, mask):
        seed = np.random.randint(2147483647)  # Random seed 생성
        torch.manual_seed(seed)  # Seed 적용
        img = self.image_transform(img)

        torch.manual_seed(seed)  # 동일한 Seed로 마스크 적용, 이미지와 동일한 변형 적용
        mask = self.mask_transform(mask)

        return img, mask

class FGADRDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask

# 데이터 변환
transform = ImageMaskTransform()

# 데이터셋 로드 및 분리
full_dataset = FGADRDataset('./data/FGADR/Original', './data/FGADR/lesionmap', transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Seed 고정
torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 모델 정의
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

# 모델, 손실함수 및 옵티마이저 설정
model = SwinUNet().cuda()

# 파라미터 고정 및 옵티마이저 설정
for name, param in model.named_parameters():
    if 'swin' in name:
        param.requires_grad = False  # 하위 층 고정

criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 학습 및 검증 루프
for epoch in range(50):
    model.train()
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        ious, dices = [], []
        for images, masks in val_loader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            predictions = predictions.squeeze(1)
            masks = masks.squeeze(1)
            predictions_np = predictions.cpu().numpy().reshape(-1).astype(int)
            masks_np = masks.cpu().numpy().reshape(-1).astype(int)
            ious.append(iou_score(masks_np, predictions_np, average='macro'))
            dices.append(f1_score(masks_np, predictions_np, average='macro'))
        print(f'Validation IoU: {np.mean(ious)}, Dice: {np.mean(dices)}')

# 모델 저장
torch.save(model.state_dict(), './model/segmentation.ckpt')