import base64
from IntelArch import get_model
from fastapi import UploadFile
from img_file import TRANSFORM
from torchvision import transforms
import numpy as np
from PIL import Image
from io import BytesIO
import torch


normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

def arr_to_buffer(img):
    img = Image.fromarray(img.astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def TransformToBase64(img: Image):
    img_np = np.array(img)
    imgR = np.zeros_like(img_np)
    imgR[:, :, 0] = img_np[:, :, 0]
    imgG = np.zeros_like(img_np)
    imgG[:, :, 1] = img_np[:, :, 1] 
    
    imgB = np.zeros_like(img_np)
    imgB[:, :, 2] = img_np[:, :, 2] 
    
    imgR_base64 = arr_to_buffer(imgR)
    imgG_base64 = arr_to_buffer(imgG)
    imgB_base64 = arr_to_buffer(imgB)
    
    return imgR_base64, imgG_base64, imgB_base64


def classify(file: UploadFile):
    model = get_model()
    device = next(model.parameters()).device
    image_bytes = file.file.read()
    
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    imgR_base64, imgG_base64, imgB_base64 = TransformToBase64(image)
    

    input_tensor = TRANSFORM(image).unsqueeze(0).to(device)
    int_to_classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    
    with torch.no_grad():
        output = model(input_tensor)

    softMax_prob = torch.softmax(output, dim=-1)

    return softMax_prob, imgR_base64, imgG_base64, imgB_base64




def apply_dropout(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    x = transform(img).unsqueeze(0) 
    x_dropped = torch.nn.functional.dropout(x, p=0.2, training=True)
    img_np = x_dropped.squeeze(0).permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np)
