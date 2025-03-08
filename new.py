from fastapi import FastAPI,File ,UploadFile,APIRouter
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
import supervision as sv
from io import BytesIO


model = YOLO("yolo11n.pt")

app = FastAPI()
main_router = APIRouter()

@app.get("/")
def root():
    return {"message": "Welcome to Demo"}

image_router = APIRouter()

@image_router.post("/detect")
async def detect_image(file : UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))


    result = model.predict(image)[0]
    detections = sv.Detections.from_ultralytics(result)


    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections)

    byte_io = BytesIO()

    annotated_image.save(byte_io, format="PNG")
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/png")

app.include_router(image_router)
