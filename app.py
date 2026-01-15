import base64
from fastapi import FastAPI, UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from img_file import FeaturesExtraction
from model import apply_dropout, classify
import os
from io import BytesIO
import PIL
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=['http://localhost:5173'],
    allow_origins=['https://cnn-explainer-frontend.vercel.app'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)
@app.get("/get")
async def testing():
    print('Got the request')   
    return {
        "message": "Hello World"
    }


@app.post("/classify")
async def get_image_pred(Img: UploadFile,):
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)
    
    
    file_content = await Img.read()
    Img.file = BytesIO(file_content)
    softMax_prob, imgR_base64,imgG_base64,imgB_base64 = classify(Img)
   
        
    return {
        "message": f"Saved {Img.filename}",
        "softMax_prob": f'{softMax_prob.tolist()}',
        "ImageR": imgR_base64,
        "ImageG": imgG_base64,
        "ImageB": imgB_base64
    }

@app.post('/getImageData')
async def getImageData(Img: UploadFile, layer: int = 0):
    file_content = await Img.read()
    feat_cls = FeaturesExtraction(file_content)
    features = feat_cls.sendFeatures_kernels()
    print('got thje features')
    return {"success":features['success'],"data":features['data'],}         

@app.post('/applyDropout')
async def applyDropout(request: Request):
    data = await request.json()  
    imgs_base64 = data["Img"]       
    results = []

    for b64 in imgs_base64:
        if b64.startswith("data:image"):
            b64 = b64.split(",")[1]
        image_bytes = base64.b64decode(b64)
        img = PIL.Image.open(BytesIO(image_bytes)).convert("RGB")  
        img_dropped = apply_dropout(img)  
        buffered = BytesIO()
        img_dropped.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        results.append(img_str)

    return {"images": results, "success": True}
   