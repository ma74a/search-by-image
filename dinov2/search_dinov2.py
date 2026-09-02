import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from feature_extractor import DINOv2FeatureExtractor
from dataset import ImageDataset
from config import Config

"""
-1 -> completely different
 0 -> unrelated
 1 -> very similar
"""

class SearchByImageDINOv2:
    def __init__(self):
        self.extractor = DINOv2FeatureExtractor()

        self.embeddings = None
        self.images_paths = []
        self.index = None

    def build_database(self):
        """Get the dataset class and dataloader
        extract the embeddings and build the index
        """
        dataset = ImageDataset(
            data_path=Config.DATA_DIR,
            transform=Config.TRANSFORM_DINOV2
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )

        print("Extracting DINOv2 embeddings...")
        self.extract_embeddings(dataloader=loader)

        print("Building FAISS index...")

        self.build_index()


    def extract_embeddings(self, dataloader):
        """extract the embeddings using extractor object and extract features function"""
        all_embeddings = []
        all_paths = []

        for images, paths in dataloader:
            features = self.extractor.extract_features(image=images)

            all_embeddings.append(features)
            all_paths.extend(paths)

        self.embeddings = torch.cat(
            all_embeddings,
            dim=0 # for each row
        )
        self.images_paths = all_paths

    def build_index(self):
        """build the index using faiss"""
        if self.embeddings is None:
            raise ValueError(
                "Extract embeddings before building the index."
            )

        features_dim = self.embeddings.shape[1] # 768
        # Build the index
        self.index = faiss.IndexFlatIP(features_dim)
        self.index.add(self.embeddings)
        print(
            f"FAISS index contains "
            f"{self.index.ntotal} images."
        )

    def search(self, query_image, top_k=5):
        """search all the embeddings and get the similarities"""
        if self.index is None:
            raise ValueError(
                "FAISS index has not been built."
            )
        img = Image.open(query_image).convert("RGB")
        img = Config.TRANSFORM_DINOV2(img).unsqueeze(0)

        query_embedding = self.extractor.extract_features(img)

        query_embedding = query_embedding.cpu().numpy()

        distance, indices = self.index.search(query_embedding, top_k+1)

        results = []
        for index, score in zip(indices[0], distance[0]):
            image_path = self.images_paths[index]

            # Skip the query image itself
            if os.path.abspath(image_path) == os.path.abspath(query_image):
                continue

            results.append({
                "path": image_path,
                "score": float(score)
            })



        return results


    def save_database(
        self,
        index_path,
        embeddings_path,
        paths_path
    ):
        """Save faiss index and features of the images and the images paths"""

        if self.index is None:
            raise ValueError(
                "FAISS index has not been built."
            )

        faiss.write_index(
            self.index,
            index_path
        )

        np.save(
            embeddings_path,
            self.embeddings
        )

        with open(paths_path, "w") as f:

            for path in self.images_paths:
                f.write(path + "\n")

        print("DINOv2 database saved.")

    def load_database(
        self,
        index_path,
        embeddings_path,
        paths_path
    ):
        """Load the index_path and features_path and images_paths"""

        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(embeddings_path)

        if not os.path.exists(paths_path):
            raise FileNotFoundError(paths_path)

        self.index = faiss.read_index(
            index_path
        )

        self.embeddings = np.load(
            embeddings_path
        )

        with open(paths_path, "r") as f:
            self.images_paths = [
                line.strip()
                for line in f
            ]

        print(
            f"Loaded database with "
            f"{self.index.ntotal} images."
        )

    def load_saved_database(
        self,
        index_path=Config.INDEX_PATH_DINO,
        embeddings_path=Config.FEATURES_PATH_DINO,
        paths_path=Config.IMAGES_PATHS_DINO
    ):
        self.load_database(
            index_path=index_path,
            embeddings_path=embeddings_path,
            paths_path=paths_path
        )
