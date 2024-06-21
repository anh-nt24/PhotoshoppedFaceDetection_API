import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageManipulationDetector:
    def __init__(self, model_path='model/model.pth'):
        self.device = torch.cuda
        self.model = self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    def _load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

    def predict(self, image):
        image = self.transform(image)
        image = image[None]
        with torch.no_grad():
            mask = self.model(image)
        heatmap = self._get_heat_map(image, mask)
        return heatmap
    
    def _get_heat_map(self, image, mask):
        return np.zeros((2, 2))