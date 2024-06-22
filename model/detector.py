import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def define_colormap():
    colors = [
        (0.0, (0, 0, 0, 0)),
        (0.1, (0, 0, 0, 0)),
        (0.2, (0, 0, 0, 0)),
        (0.3, (102/255, 205/255, 170/255, 0.0)),
        (0.4, (102/255, 205/255, 170/255, 0.0)),
        (0.5, (0/255, 0/255, 255/255, 0.5)),
        (0.6, (102/255, 205/255, 170/255, 0.6)),
        (0.65, (102/255, 205/255, 170/255, 0.7)),
        (0.7, (255/255, 255/255, 0/255, 0.3)),       
        (0.8, (255/255, 255/255, 0/255, 0.5)),
        (1.0, 'red')
    ]  
    cmap_name = 'transparent_cyan_blue_yellow_red'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    return custom_cmap

class ImageManipulationDetector:
    def __init__(self, base_model, model_path='model/v1/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, base_model)
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self.color_map = define_colormap()
        
    def _load_model(self, model_path, base_model):
        base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        base_model.to(self.device)
        base_model.eval()
        return base_model

    def predict(self, image):
        try:
            image_tf = self.transform(image)
            image_tf = image_tf[None]
            image_tf = image_tf.to(self.device)
            with torch.no_grad():
                mask = self.model(image_tf)
            heatmap = self._get_heat_map(image, mask)
            return heatmap
        except Exception as e:
            raise RuntimeError(f'Error during prediction: {str(e)}')
    
    
    def _get_heat_map(self, image, mask):
        original_size = image.size
        # normalize mask
        mask = np.squeeze(mask)
        mask_min, mask_max = mask.min(), mask.max()
        normalized_mask = (mask - mask_min) / (mask_max - mask_min)
        
        # get heatmap
        heatmap = self.color_map(normalized_mask.cpu().numpy())
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = Image.fromarray(heatmap)
        unresized_heatmap = heatmap.resize(original_size, Image.BICUBIC)
        
        # overlay the heatmap on the input image
        image.paste(unresized_heatmap, (0,0), mask = unresized_heatmap)

        return image