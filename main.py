from fastapi import FastAPI,HTTPException,File,UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
app=FastAPI(title="Image-classifier-api")

def image_preprocess(image_bytes:bytes):
               image=Image.open(image_bytes)
               image.resize(32,32)
               image=np.array(image)/255.0
               image=np.expand_dims(image,axis=0)
               return image

def predict_image(image_bytes:bytes):
    """
    A placeholder prediction function.
    In a real-world scenario, load your trained model and process `image_bytes` to extract features and predict a label.
    """
    return{"class":"cat","confidence":0.95}

@app.get("/predict")
async def predict(file:UploadFile=File(...)):
               if not file.content_type.startswith("image/"):
                              raise HTTPException(status_code=400,detail="Upload file is not image")
               image_bytes=await file.read()
               
               result=predict_image(image_bytes)
               
               return JSONResponse(result)


if __name__=="__main__":
               uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)