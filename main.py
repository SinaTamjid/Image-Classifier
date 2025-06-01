from fastapi import FastAPI,HTTPException,File,UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
from keras.models import load_model
import io
model=load_model(r"Image-Classifier/model.keras")


app=FastAPI(title="Image-classifier-api")

def image_preprocess(image_bytes:bytes):
               image=Image.open(io.BytesIO(image_bytes))
               image = image.resize((32, 32), Image.Resampling.BILINEAR)
               image=np.array(image)/255.0
               image=np.expand_dims(image,axis=0)
               return image

def predict_image(image_bytes:bytes):
    """
    A placeholder prediction function.
    In a real-world scenario, load your trained model and process `image_bytes` to extract features and predict a label.
    """
    image=image_preprocess(image_bytes)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)
    confidence=float(np.max(prediction))
    return{"class":int(predicted_class),"confidence":confidence}

@app.post("/predict")
async def predict(file:UploadFile=File(None)):
               if file is None:
                              raise HTTPException(status_code=400, detail="A file is required for prediction")
               if not file.content_type.startswith("image/"):
                              raise HTTPException(status_code=400,detail="Upload file is not image")
               image_bytes=await file.read()
               
               result=predict_image(image_bytes)
               
               return JSONResponse(result)



if __name__=="__main__":
               uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)