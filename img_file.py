import BytesIO
import PIL
import numpy as np

def convertImg_to_arr(file_content):
    img = PIL.Image.open(BytesIO(file_content))       
    img_arr = np.array(img)                      
    return img_arr.shape, img_arr