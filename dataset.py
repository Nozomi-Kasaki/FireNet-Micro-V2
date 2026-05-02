import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_train_transforms(img_size=224):
    resize_size = max(256, int(round(img_size * 256 / 224)))
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=224):
    resize_size = max(256, int(round(img_size * 256 / 224)))
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name
