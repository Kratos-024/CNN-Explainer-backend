from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from img_file import FeaturesExtraction
from model import classify
import os
from io import BytesIO
import PIL
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/classify")
async def get_image_pred(Img: UploadFile,):
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)
    
    
    file_content = await Img.read()
    Img.file = BytesIO(file_content)
    predicted_class, imgR_base64,imgG_base64,imgB_base64 = classify(Img)
   
        
    return {
        "message": f"Saved {Img.filename}",
        "predicted_class": predicted_class,
        "ImageR": imgR_base64,
        "ImageG": imgG_base64,
        "ImageB": imgB_base64
    }

@app.post('/getImageData')
async def getImageData(Img: UploadFile, layer: int = 0):
    file_content = await Img.read()
    feat_ext = FeaturesExtraction(file_content)
    features = feat_ext.sendFeatures(layer)
    print('got thje features')
    return {"success":features['success'],"first_conv_images":features['first_conv_images'],"first_relu_images":features['first_relu_images']}         

    

# @app.post('/getFeatureMapsImage')
# async def getFeatureMapsImage(Img: UploadFile, layer: int = 0):
#     file_content = await Img.read()
#     feat_ext = FeaturesExtraction(file_content)
#     features = feat_ext.sendFeatures_kernels(layer)
#     print(features)
#     return {"shape": features.shape, "features": features.tolist()}
