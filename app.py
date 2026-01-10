from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import classify
import os
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/")
async def get_image(Img: UploadFile,):
    print("oijfsdjodfdjofdosf")
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)
    
    
    file_content = await Img.read()
    file_path = os.path.join(upload_dir, Img.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
    
    Img.file = BytesIO(file_content)
    
    
    predicted_class, imgR_base64,imgG_base64,imgB_base64 = classify(Img)
   
        
    return {
        "message": f"Saved {Img.filename}",
        "predicted_class": predicted_class,
        "ImageR": imgR_base64,
        "ImageG": imgG_base64,
        "ImageB": imgB_base64
    }
