import torch
import torch.nn.functional as F


from config import Config

class DINOv2FeatureExtractor:
    def __init__(self):
        self.model = torch.hub.load(
        'facebookresearch/dinov2',
        'dinov2_vitb14'
        )
        self.model.eval()

        self.tranforms = Config.TRANSFORM
        self.features = []
        self.images_path = []

    def extract_features(self, image):

        with torch.no_grad():
            features = self.model(image) # torch.Size([1, 768])

        features = F.normalize(
            features,
            p=2,
            dim=1
        )

        # features = features.squeeze(0) # torch.Size([768])

        return features