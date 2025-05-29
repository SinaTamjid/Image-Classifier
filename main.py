from fastapi import FastAPI,HTTPException
import uvicorn
app=FastAPI(title="Image-classifier-api")

@app.get("/predict/")
def predict():
               pass
















if __name__=="__main__":
               uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)