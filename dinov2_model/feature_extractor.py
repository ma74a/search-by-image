import torch
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from PIL import Image


class DINOv2FeatureExtractor:
    def __init__(self):
        self.model = torch.hub.load(
        'facebookresearch/dinov2',
        'dinov2_vitb14'
        )
        self.model.eval()


    def extract_features(self, image):

        with torch.no_grad():
            outputs = self.model.forward_features(image)
            features = outputs['x_norm_clstoken']  # (1, 768) — global representation

        features = F.normalize(
            features,
            p=2,
            dim=1
        )

        # features = features.squeeze(0) # torch.Size([768])

        return features
