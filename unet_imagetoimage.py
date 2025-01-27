import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import functional as F
import random

# Directories
input_dir = "sample_images_unet" 
output_dir = "sample_images_unet_generated"
os.makedirs(output_dir, exist_ok=True)

# Data Augmentation Pipeline
class AugmentedDataset(Dataset):
    def __init__(self, input_dir, augmentations, image_size=(1600, 1200)):
        self.input_dir = input_dir
        self.image_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        self.augmentations = augmentations
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        augmented_image = self.augmentations(image)
        return self.transform(image), self.transform(augmented_image)

# Augmentation pipeline
def augment_image(image):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=(1600, 1200), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    augmented = augmentations(image)
    return augmented

# U-Net Model for Image-to-Image Translation
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = up_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.up2 = up_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.up1 = up_block(128, 64)
        self.decoder1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

# Training Loop
def train_model(model, dataloader, num_epochs=50, learning_rate=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for original, augmented in dataloader:
            original, augmented = original.to(device), augmented.to(device)
            optimizer.zero_grad()
            output = model(augmented)
            loss = criterion(output, original)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)}")

# Data Preparation
augmentations = lambda x: augment_image(x)
dataset = AugmentedDataset(input_dir, augmentations)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model Initialization and Training
unet_model = UNet()
train_model(unet_model, dataloader)

# Save the model
torch.save(unet_model.state_dict(), "unet_image_to_image.pth")

print(f"Model training complete. Saved to 'unet_image_to_image.pth'.")
