from io import BytesIO
import PIL
import numpy as np
from torchvision import transforms
from IntelArch import get_model
import torch
import matplotlib.cm as cm 
import base64

model = get_model()
model.eval()

normalize_mean = [0.485, 0.456, 0.406]
normalize_std  = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

class FeaturesExtraction:
    def __init__(self, file_content):
        self.file_content = file_content
        self.img_arr = PIL.Image.open(BytesIO(file_content)).convert("RGB")
        input_tensor = TRANSFORM(self.img_arr)
        self.input_tensor = input_tensor.unsqueeze(0)

    def convertImg_to_arr(self): 
        img_arr = np.array(self.img_arr)      
        return img_arr.shape, img_arr
                        
    def sendFeatures_kernels(self):
        target_indices = range(20) 

        captured_features = {} 
        x = self.input_tensor
   
        all_layers = []
        all_layers.extend(list(model.conv1.children()))
        all_layers.extend(list(model.conv2.children()))
        all_layers.extend(list(model.conv3.children()))
        all_layers.extend(list(model.conv4.children()))
   
            
        for i, layer in enumerate(all_layers):
            x = layer(x)
                
            if i in target_indices:
                output_tensor = x.detach().cpu()
                channel_scores = output_tensor[0].view(output_tensor.shape[1], -1).sum(dim=1)
                k = min(10, len(channel_scores))
                _, top_indices = torch.topk(channel_scores, k=k)
                
                selected_maps = output_tensor[0][top_indices].numpy()

                processed_maps = []
                for feature in selected_maps:
                    min_val = feature.min()
                    max_val = feature.max()
                        
                    if max_val - min_val > 0:
                        norm_feature = (feature - min_val) / (max_val - min_val + 1e-5)
                        norm_feature = (norm_feature * 255).astype(np.uint8)
                    else:
                        norm_feature = feature.astype(np.uint8)
                            
                    processed_maps.append(self.arr_to_buffer(norm_feature))

                captured_features[f"layer_{i}"] = processed_maps
                    
            if i >= 20:
                break

        return {
                'success': True,
                'data': captured_features
            }

    def arr_to_buffer(self, img_arr):
            if img_arr.dtype == np.uint8:
                norm_arr = img_arr / 255.0
            else:
                norm_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-5)
            colored_map = cm.viridis(norm_arr) 
            
            colored_map = (colored_map * 255).astype(np.uint8)
            im = PIL.Image.fromarray(colored_map)
            
            buffer = BytesIO()
            im.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return img_str