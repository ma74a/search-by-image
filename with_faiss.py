from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torch

from PIL import Image
import numpy as np
import faiss
import os

from dataset import ImageDataset
from config import Config

class SearchByImage:
    def __init__(self):
        """Iniliaze the model(resnet50) and 
        """
        self.model = resnet50(weights="IMAGENET1K_V1")
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.transform = Config.TRANSFORM
        self.image_paths = []
        self.features = []
        self.index = None
        
    def extract_features(self, dataloader):
        """Extract the featues of the images using resnet50

        Args:
            dataloader (torch.utils.data.DataLoader): 
                A DataLoader providing image batches and their associated labels or paths.
        """
        get_features = []
        paths = []
        
        with torch.no_grad():
            for img, img_path in dataloader:
                # We use reshape to keep the batch dim
                batch_features = self.model(img).numpy().reshape(img.shape[0], -1)
                get_features.append(batch_features)
                paths.extend(img_path)
        
        
        self.features = np.vstack(get_features)
        self.image_paths = paths
    
    def build_faiss_index(self):
        """Build the faiss index for the data using the featutes
        """
        dataset = ImageDataset(data_path=Config.DATA_DIR,
                               transform=self.transform)
        
        dataloader = DataLoader(dataset=dataset,
                                batch_size=Config.BATCH_SIZE,
                                drop_last=True)
        
        self.extract_features(dataloader)
        features_dim = self.features.shape[1]
        
        # Normalize features
        faiss.normalize_L2(self.features)
        # build the index
        self.index = faiss.IndexFlatIP(features_dim)
        self.index.add(self.features)
        
    def search(self, query_path, top_k=5):
        """Search and retrive most similar images

        Args:
            query_path (str): The path of the query image
            top_k (int): The most similar images. Defaults to 5.

        Returns:
            list: List of all similar images of len k
        """
        img = Image.open(query_path).convert("RGB")
        img = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(img).numpy().reshape(1, -1)
            
        faiss.normalize_L2(output)
        distance, indices = self.index.search(output, top_k)
        
        similars = [self.image_paths[i] for i in indices[0]]
        
        return similars
        
        
    def save_index_features(self, index_path, features_path):
        """Save faiss index and features of the images

        Args:
            index_path (str): Path for the saved index
            features_path (str): Path for the save features
        """
        faiss.write_index(self.index, index_path)
        
        np.save(features_path, self.features)
        
    def save_image_paths(self, img_paths):
        """Save Image paths to use them

        Args:
            img_paths (str): the path to save all the images paths
        """
        with open(img_paths, "w") as f:
            for path in self.image_paths:
                f.write(f"{path}\n")
        print(f"Image paths saved as {img_paths}")
    
    def load_index_features(self, index_path, features_path):
        """Load the index_path and features_path

        Args:
            index_path (str): Path for the saved index
            features_path (str): Path for the save features
        """
        # Load index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file {index_path} not found")
        self.index = faiss.read_index(index_path)
        # print(f"Loaded index from {index_path}")
                
        # Load Features
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file {features_path} not found")
        self.features = np.load(features_path)
        # print(f"Loaded {len(self.features)} feature vectors from {features_path}")
    
    def load_image_paths(self, paths_file):
        """Load the saved image paths."""

        if not os.path.exists(paths_file):
            raise FileNotFoundError(f"Image paths file {paths_file} not found")
        
        with open(paths_file, "r") as f:
            self.image_paths = [line.strip() for line in f]
        # print(f"Loaded {len(self.image_paths)} image paths from {paths_file}")
    
    def load_save_model(self, index_path=Config.INDEX_PATH,
                        features_path=Config.FEATURES_PATH, 
                        img_paths=Config.IMAGES_PATHS):
        """Load the index_path, features_path and img_path

        Args:
            index_path (str)
            features_path (str)
            img_paths (str)
        """
        self.load_index_features(index_path=index_path, features_path=features_path)
        self.load_image_paths(paths_file=img_paths)
        

if __name__ == "__main__":
    obj = SearchByImage()
    # obj.build_faiss_index()
    # obj.save_index_features(index_path=Config.INDEX_PATH, features_path=Config.FEATURES_PATH)
    # obj.save_image_paths(img_paths=Config.IMAGES_PATHS)
    
    similars = obj.search("data/the_images/٢ سرير.jpg")
    for img in similars:
        print(similars)
