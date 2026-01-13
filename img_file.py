from io import BytesIO
import PIL
from model import arr_to_buffer
import numpy as np
from torchvision import transforms
from IntelArch import get_model
import torch
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
                
        target_indices = [0, 2, 5, 7, 8, 10, 12, 13, 15, 17, 18]
        captured_features = {} 
        x = self.input_tensor
        for i, layer in enumerate(model.features.children()):
            x = layer(x)
            if i in target_indices:
                output_tensor = x.detach().cpu()
                channel_scores = output_tensor[0].view(output_tensor.shape[1], -1).sum(dim=1)
                _, top_indices = torch.topk(channel_scores, k=10)
                selected_maps = output_tensor[0][top_indices].numpy()
                captured_features[f"layer_{i}"] = [arr_to_buffer(feature) for feature in selected_maps]
                if i == 18:
                    break
       


        return {
            'success': True,
             'data': captured_features}