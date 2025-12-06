from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import io
import base64
from inference import classifier
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources on startup
    success = classifier.load_resources()
    if not success:
        print("Failed to load resources on startup.")
    yield
    # Clean up resources on shutdown if needed
    pass

app = FastAPI(lifespan=lifespan)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "Bird Classifier API"}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    start: float = Form(0.0),
    duration: float = Form(None)
):
    if not (file.content_type.startswith("audio/") or file.content_type.startswith("video/")):
        pass 
    
    try:
        print(f"API DEBUG: Received predict request with start={start}, duration={duration}")
        contents = await file.read()
        results, images = classifier.predict(contents, filename=file.filename, offset=start, duration=duration)
        
        if results is None:
            raise HTTPException(status_code=500, detail="Inference failed")
            
        spectrograms = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            spectrograms.append(f"data:image/png;base64,{img_str}")
        
        return JSONResponse(content={
            "predictions": results,
            "spectrograms": spectrograms
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
