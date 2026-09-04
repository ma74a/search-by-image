import os

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from dataset import ImageDataset
from dinov2_model.feature_extractor import DINOv2FeatureExtractor


class SearchByImageFusion:
    """
    Image search using a combination of:
        - ResNet50: 2048-dimensional embedding
        - DINOv2 ViT-B/14: 768-dimensional embedding

    Combined embedding:
        2048 + 768 = 2816 dimensions
    """

    def __init__(self):


        # ResNet50
        self.resnet = resnet50(
            weights="IMAGENET1K_V1"
        )

        # Remove the final classification layer
        self.resnet = torch.nn.Sequential(
            *list(self.resnet.children())[:-1]
        )

        self.resnet = self.resnet
        self.resnet.eval()

        # DINOv2

        self.dino = DINOv2FeatureExtractor()

        # Image preprocessing

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Database

        self.embeddings = None
        self.image_paths = []
        self.index = None


    def extract_resnet(self, images):
        """
        Extract ResNet50 embeddings.

        Input:
            [B, 3, 224, 224]

        Output:
            [B, 2048]
        """

        images = images

        with torch.no_grad():

            features = self.resnet(images)

            # [B, 2048, 1, 1] -> [B, 2048]
            features = features.flatten(1)

            # Normalize ResNet embedding
            features = F.normalize(
                features,
                p=2,
                dim=1
            )

        return features
    
    # DINOv2 embeddin
    def extract_dino(self, images):
        """
        Extract DINOv2 embeddings.

        Input:
            [B, 3, 224, 224]

        Output:
            [B, 768]
        """

        features = self.dino.extract_features(images)

        return features
    
    # Combine ResNet + DINOv
    def combine_embeddings(
        self,
        resnet_embeddings,
        dino_embeddings
    ):
        """
        Combine normalized ResNet and DINOv2 embeddings.

        ResNet:
            [B, 2048]

        DINOv2:
            [B, 768]

        Combined:
            [B, 2816]
        """

        # Make sure both embeddings are normalized
        resnet_embeddings = F.normalize(
            resnet_embeddings,
            p=2,
            dim=1
        )

        dino_embeddings = F.normalize(
            dino_embeddings,
            p=2,
            dim=1
        )

        # Concatenate the two representations
        combined = torch.cat(
            [
                resnet_embeddings,
                dino_embeddings
            ],
            dim=1
        )

        # Normalize the final representation
        combined = F.normalize(
            combined,
            p=2,
            dim=1
        )

        return combined
    
    # Extract combined embeddin
    def extract_embedding(self, images):
        """
        Extract the combined ResNet + DINOv2 embedding.

        Input:
            [B, 3, 224, 224]

        Output:
            [B, 2816]
        """

        resnet_embeddings = self.extract_resnet(
            images
        )

        dino_embeddings = self.extract_dino(
            images
        )

        combined_embeddings = self.combine_embeddings(
            resnet_embeddings,
            dino_embeddings
        )

        return combined_embeddings
    
    # Extract database embedding
    def extract_database_embeddings(self, dataloader):
        """
        Extract combined embeddings for all database images.
        """

        all_embeddings = []
        all_paths = []

        for images, paths in dataloader:

            embeddings = self.extract_embedding(
                images
            )

            all_embeddings.append(
                embeddings.cpu()
            )

            all_paths.extend(paths)

        # [B, 2816] + [B, 2816] + ...
        #          ↓
        # [N, 2816]

        self.embeddings = torch.cat(
            all_embeddings,
            dim=0
        ).numpy()

        self.image_paths = all_paths

        print(
            f"Extracted embeddings shape: "
            f"{self.embeddings.shape}"
        )

    # Build FAISS index
    def build_index(self):
        """
        Build FAISS Inner Product index.

        Since the embeddings are L2-normalized,
        Inner Product is equivalent to cosine similarity.
        """

        if self.embeddings is None:
            raise ValueError(
                "Embeddings have not been extracted."
            )

        dimension = self.embeddings.shape[1]

        print(
            f"Embedding dimension: {dimension}"
        )

        # Normalize once more at the NumPy level
        # to guarantee FAISS receives normalized vectors.
        faiss.normalize_L2(
            self.embeddings
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            self.embeddings
        )

        print(
            f"FAISS index contains "
            f"{self.index.ntotal} images."
        )

    # Build complete databas
    def build_database(
        self,
        data_path,
        batch_size=32
    ):
        """
        Complete database-building pipeline:

            Dataset
                ↓
            DataLoader
                ↓
            ResNet + DINOv2
                ↓
            2816-D embeddings
                ↓
            FAISS
        """

        dataset = ImageDataset(
            data_path=data_path,
            transform=self.transform
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        print(
            "Extracting ResNet + DINOv2 embeddings..."
        )

        self.extract_database_embeddings(
            dataloader
        )

        print(
            "Building combined FAISS index..."
        )

        self.build_index()

    # Search
    def search(
        self,
        query_path,
        top_k=5
    ):
        """
        Search for images similar to the query.

        Returns:
            [
                {
                    "path": image_path,
                    "score": similarity_score
                },
                ...
            ]
        """

        if self.index is None:
            raise ValueError(
                "FAISS index has not been built."
            )

        # ------------------------------------------
        # Load query image
        # ------------------------------------------

        image = Image.open(
            query_path
        ).convert("RGB")

        # ------------------------------------------
        # Transform
        # ------------------------------------------

        image = self.transform(image)

        # [3, 224, 224] -> [1, 3, 224, 224]
        image = image.unsqueeze(0)

        # ------------------------------------------
        # Extract combined embedding
        # ------------------------------------------

        query_embedding = self.extract_embedding(
            image
        )

        query_embedding = (
            query_embedding
            .cpu()
            .numpy()
        )

        # ------------------------------------------
        # Search
        # ------------------------------------------

        # +1 because the query itself may be
        # present in the database.
        distances, indices = self.index.search(
            query_embedding,
            top_k + 1
        )

        # ------------------------------------------
        # Build results
        # ------------------------------------------

        results = []

        query_path_abs = os.path.abspath(
            query_path
        )

        for index, score in zip(
            indices[0],
            distances[0]
        ):

            image_path = self.image_paths[index]

            # Don't return the query itself
            if (
                os.path.abspath(image_path)
                == query_path_abs
            ):
                continue

            results.append({
                "path": image_path,
                "score": float(score)
            })

            if len(results) == top_k:
                break

        return results
    
    # Save databas
    def save_database(
        self,
        index_path,
        embeddings_path,
        paths_path
    ):
        """
        Save:

            1. FAISS index
            2. Embeddings
            3. Image paths
        """

        if self.index is None:
            raise ValueError(
                "FAISS index has not been built."
            )

        # ------------------------------------------
        # Save FAISS index
        # ------------------------------------------

        faiss.write_index(
            self.index,
            index_path
        )

        # ------------------------------------------
        # Save embeddings
        # ------------------------------------------

        np.save(
            embeddings_path,
            self.embeddings
        )

        # ------------------------------------------
        # Save image paths
        # ------------------------------------------

        with open(paths_path, "w") as f:

            for path in self.image_paths:
                f.write(f"{path}\n")

        print(
            "Combined ResNet + DINOv2 database saved."
        )

    # Load databas
    def load_database(
        self,
        index_path="saved/fusion.faiss",
        embeddings_path="saved/fusion_embeddings.npy",
        paths_path="saved/fusion_paths.txt"
    ):
        """
        Load an existing database without
        extracting embeddings again.
        """

        # ------------------------------------------
        # Check files
        # ------------------------------------------

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found: {index_path}"
            )

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings not found: "
                f"{embeddings_path}"
            )

        if not os.path.exists(paths_path):
            raise FileNotFoundError(
                f"Paths file not found: {paths_path}"
            )

        # ------------------------------------------
        # Load FAISS
        # ------------------------------------------

        self.index = faiss.read_index(
            index_path
        )

        # ------------------------------------------
        # Load embeddings
        # ------------------------------------------

        self.embeddings = np.load(
            embeddings_path
        )

        # ------------------------------------------
        # Load image paths
        # ------------------------------------------

        with open(paths_path, "r") as f:

            self.image_paths = [
                line.strip()
                for line in f
            ]

        print(
            f"Loaded database with "
            f"{self.index.ntotal} images."
        )


