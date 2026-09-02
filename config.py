import torch
from torchvision import transforms

class Config:
    # Path
    DATA_DIR = "data/the_images"

    INDEX_PATH_RESNET = "saved/faiss_index_resnet.faiss"
    FEATURES_PATH_RESNET = "saved/features_resnet.npy"
    IMAGES_PATHS_RESNET = "saved/images_paths_resnet.txt"

    INDEX_PATH_DINO = "saved/faiss_index_dinov2.faiss"
    FEATURES_PATH_DINO = "saved/features_dinov2.npy"
    IMAGES_PATHS_DINO = "saved/images_paths_dinov2.txt"
    
    # Parameters
    BATCH_SIZE = 32
    IMG_SIZE = 224
    
    # Image preprocessing
    TRANSFORM_RESNET = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    TRANSFORM_DINOV2 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),   # 224 = 14 × 16 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
])
    
    
    