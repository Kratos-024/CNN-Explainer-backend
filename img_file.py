from io import BytesIO
import PIL
from model import arr_to_buffer
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

    # def sendFeatures_kernels(self, layer_index=0): 
    #     first_layer = list(model.features.children())[layer_index] 
    #     first_relu = list(model.features.children())[layer_index+2]  
    #     print('list(model.features.children())',len(list(model.features.children())))
    #     print('list(model.features.children())',len(list(model.classifier.children())))
        
        
    #     first_convOutput = first_layer(self.input_tensor)
    #     first_reluOutput = first_relu(first_convOutput) 
        
    #     first_convOutput = first_convOutput.detach().cpu().numpy()[0][0:10]
    #     first_convOutput_base64 = [arr_to_buffer(feature) for feature in first_convOutput]
        
    #     first_reluOutput = first_reluOutput.detach().cpu().numpy()[0][0:10]
    #     first_reluOutput_base64 = [arr_to_buffer(feature) for feature in first_reluOutput]

    #     return {
    #         'success': True,
    #         'firstConvLayer': first_convOutput_base64,  
    #         'firstReluLayer': first_reluOutput_base64  
    #     }

                        
    def sendFeatures_kernels(self):
         
        # target_indices = [0, 2, 5, 7, 8, 10, 12, 13, 15, 17, 18]
        target_indices = [0, 1, 2, 3,4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18]

        captured_features = {} 
        x = self.input_tensor
            
        for i, layer in enumerate(model.features.children()):
            x = layer(x)
                
            if i in target_indices:
                output_tensor = x.detach().cpu()
                    
                  
                channel_scores = output_tensor[0].view(output_tensor.shape[1], -1).sum(dim=1)
                _, top_indices = torch.topk(channel_scores, k=10)
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
                    
                if i == 18:
                        break

        return {
                'success': True,
                'data': captured_features
            }

    def arr_to_buffer(self,img_arr):
            # 1. Normalize to 0-1 range (Matplotlib expects floats 0.0 - 1.0)
            # img_arr is likely uint8 (0-255), so divide by 255
            if img_arr.dtype == np.uint8:
                norm_arr = img_arr / 255.0
            else:
                norm_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-5)

            # 2. Apply Colormap (e.g., 'viridis', 'plasma', 'inferno', 'magma')
            # This turns (H, W) -> (H, W, 4) (RGBA)
            colored_map = cm.viridis(norm_arr) 
            
            # 3. Convert to 0-255 integers
            colored_map = (colored_map * 255).astype(np.uint8)
            
            # 4. Create Image (Mode is now 'RGBA' because of the colormap)
            im = PIL.Image.fromarray(colored_map)
            
            # 5. Save to buffer
            buffer = BytesIO()
            im.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return img_str
