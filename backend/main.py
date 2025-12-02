from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
from inference import predict, load_resources

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources on startup
@app.on_event("startup")
async def startup_event():
    load_resources()

@app.get("/")
def read_root():
    return {"Hello": "Bird Classifier API"}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    start: float = 0.0,
    duration: float = None
):
    if not file.content_type.startswith("audio/"):
        # Allow video files too as per logic, but warn or check extension
        pass 
    
    try:
        contents = await file.read()
        results, image = predict(contents, filename=file.filename, offset=start, duration=duration)
        
        if results is None:
            raise HTTPException(status_code=500, detail="Inference failed")
            
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse(content={
            "predictions": results,
            "spectrogram": f"data:image/png;base64,{img_str}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
