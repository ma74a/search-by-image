from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image

class ImageDataset(Dataset):
    """Custom dataset for loading and preprocessing images."""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.image_paths = self._get_paths()
        
    def _get_paths(self):
        """Get the paths of the images"""
        img_paths = []
        for img in os.listdir(self.data_path):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.data_path, img)
                img_paths.append(img_path)
            
        return img_paths
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Return the image and its path"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_path
        