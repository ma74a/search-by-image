import torch
from torchvision import transforms

class Config:
    # Path
    DATA_DIR = "data/the_images"
    INDEX_PATH = "saved/faiss_index.faiss"
    FEATURES_PATH = "saved/features.npy"
    IMAGES_PATHS = "saved/images_paths.txt"
    
    # Parameters
    BATCH_SIZE = 32
    IMG_SIZE = 224
    
    # Image preprocessing
    TRANSFORM = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    
    