from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image
import io
import requests
import cv2
import numpy as np
import uuid
import os
from dotenv import load_dotenv, find_dotenv
import boto3
import logging

app = FastAPI()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the key.env file
dotenv_path = find_dotenv('key.env')
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError('The key.env file was not found')

# Load environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION')
bucket_name = os.getenv('S3_BUCKET_NAME')

# Create S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_default_region
)

# Model path
model_path = 'runs/best.pt'
logger.info(f"Model path: {model_path}")

# Load YOLOv5 model using torch.hub.load
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

class Prediction(BaseModel):
    foodName: str
    accuracy: float

class ResponseModel(BaseModel):
    predictions: list[Prediction]
    image_url: str

@app.post("/predict", response_model=ResponseModel)
async def predict(request: Request):
    try:
        request_data = await request.json()
        logger.info(f"Received request data: {request_data}")

        image_url = request_data.get("image_url")
        if not image_url:
            raise HTTPException(status_code=400, detail="No image URL provided")

        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")

        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        # YOLOv5 prediction
        results = model(img)
        predictions = []

        # Convert image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Draw bounding boxes for each prediction
        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(opencv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Add prediction results
            predictions.append(Prediction(
                foodName=model.names[int(cls)],
                accuracy=conf
            ))

        # Save the image with bounding boxes to memory
        _, buffer = cv2.imencode('.jpg', opencv_image)
        buffer = io.BytesIO(buffer)

        # Upload image to S3
        s3_key = f"peanut/result/{uuid.uuid4().hex}.jpg"
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        image_url = f"https://{bucket_name}.s3.{aws_default_region}.amazonaws.com/{s3_key}"

        # Build response
        response = ResponseModel(predictions=predictions, image_url=image_url)
        logger.info(f"Response data: {response.dict()}")

        return JSONResponse(content=response.dict())

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error running server: {e}")
