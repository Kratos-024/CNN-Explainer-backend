from io import BytesIO
import PIL
from model import arr_to_buffer
import numpy as np
from torchvision import transforms
from IntelArch import get_model

model = get_model()
model.eval()

normalize_mean = [0.485, 0.456, 0.406]
normalize_std  = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean,std=normalize_std)
])

class FeaturesExtraction:
    def __init__(self, file_content):
        self.file_content = file_content
        self.img_arr = PIL.Image.open(BytesIO(file_content))
        input_tensor = TRANSFORM(self.img_arr)
        self.input_tensor = input_tensor.unsqueeze(0)

    def convertImg_to_arr(self): 
        img_arr = np.array(self.img_arr)      
        return img_arr.shape, img_arr

    def sendFeatures_kernels(self, layer_index=0): 
        first_layer = list(model.features.children())[layer_index]  
        features = first_layer(self.input_tensor)
        features = features.detach().cpu().numpy()[0][0:10]
        features_base64 = [arr_to_buffer(feature) for feature in features]
        return np.array(features_base64)
    

        
