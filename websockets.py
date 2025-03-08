from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import base64
import supervision as sv


model = YOLO("yolo11n.pt")


app = FastAPI()


image_router = APIRouter()

@app.get("/")
def root():
    return {"message": "Welcome to Demo"}

app.include_router(image_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@image_router.websocket("/detect")
async def detect_image(websocket: WebSocket, url: str):
    
    await websocket.accept()

    cap = cv2.VideoCapture(url)  

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  

            
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            
            result = model.predict(pil_image, conf=0.25)[0]  

            detections = sv.Detections.from_ultralytics(result)

          
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections)

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            byte_data = buffer.tobytes()
            frame_base64 = base64.b64encode(byte_data).decode('utf-8')


            await websocket.send_text(frame_base64)

    except WebSocketDisconnect:
        print("Client disconnected")
    
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

    finally:
        cap.release()

