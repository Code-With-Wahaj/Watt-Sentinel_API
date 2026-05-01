# api.py
import cv2
import numpy as np
import socket
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException # Added HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO # Ensure ultralytics is installed: pip install ultralytics
from collections import Counter # Import Counter for counting labels

# --- Configuration ---
# Replace with the actual path to your trained YOLO model
MODEL_PATH = r"C:\Users\jbss\Desktop\pt model\best.pt"
# Host configuration: "0.0.0.0" makes the server accessible on your network
HOST_IP = "0.0.0.0"
PORT = 8000

# --- FastAPI App Initialization ---
app = FastAPI(title="YOLO Object Detection API with Summary")

# --- CORS Middleware ---
# Allows requests from your Flutter web app.
# IMPORTANT: For production, replace "*" with your specific Flutter app's URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["POST"], # Allow only POST requests
    allow_headers=["*"], # Allow all headers for simplicity during testing
)

# --- Load YOLO Model ---
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO model loaded successfully from: {MODEL_PATH}")
    print(f"ℹ️ Model Classes: {model.names}") # Print class names on startup
except Exception as e:
    print(f"❌ Fatal Error: Could not load YOLO model from {MODEL_PATH}. Exception: {e}")
    # Exit if the model can't load, as the API is useless without it.
    exit(1) # Use exit code 1 to indicate an error

# --- Helper Function to Get Local IP ---
def get_local_ip():
    """Automatically get the machine's local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.1) # Prevent hanging indefinitely
    try:
        # Doesn't actually send data, just connects to get local IP
        s.connect(("8.8.8.8", 80)) # Connect to Google DNS
        ip = s.getsockname()[0]
    except Exception:
        try:
            # Fallback: Get IP from hostname
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
             ip = "127.0.0.1" # Final fallback to localhost
    finally:
        s.close()
    return ip

# --- API Endpoint ---
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Receives an image file, performs YOLO detection, counts detected objects,
    and returns both the detailed detections and a summary count.
    """
    try:
        # 1. Read image bytes from the uploaded file
        image_bytes = await file.read()
        if not image_bytes:
             raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # 2. Decode image bytes into an OpenCV image format
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            # Use HTTPException for standard FastAPI error responses
            raise HTTPException(status_code=400, detail="Failed to decode image. Ensure it's a valid image format (JPG, PNG, etc.).")

        # 3. Run YOLOv8 detection
        results = model(image, verbose=False) # verbose=False reduces console spam

        # 4. Process and format detections
        detections = []
        label_list = [] # List to store just the labels for counting
        if results and results[0].boxes: # Check if results and boxes exist
            boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes [x1, y1, x2, y2]
            confs = results[0].boxes.conf.cpu().numpy() # Confidence scores
            clss = results[0].boxes.cls.cpu().numpy()   # Class IDs

            for box, conf, cls_id in zip(boxes, confs, clss):
                # Ensure class ID is valid
                if int(cls_id) in model.names:
                    class_name = model.names[int(cls_id)] # Get class name from model
                    detections.append({
                        "box": box.tolist(),       # Convert numpy array to list [x1, y1, x2, y2]
                        "label": class_name,       # Class name (e.g., 'person', 'car')
                        "confidence": float(conf)  # Confidence score as float
                    })
                    label_list.append(class_name) # Add label to list for counting
                else:
                    print(f"⚠️ Warning: Detected unknown class ID: {int(cls_id)}")


        # 5. Count detected objects by label --- NEW ---
        # Use collections.Counter to efficiently count items in the list
        detection_summary = Counter(label_list)

        # 6. Return detections and the summary count --- UPDATED ---
        return {
            "detections": detections,
            "detection_summary": dict(detection_summary) # Convert Counter to dict for JSON compatibility
        }

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise http_exc
    except Exception as e:
        print(f"❌ Error during detection processing: {e}")
        # Return a generic server error response
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


# --- Server Startup ---
if __name__ == "__main__":
    local_ip = get_local_ip()
    print("\n--- YOLO Detection Server (with Summary) ---")
    print(f"✨ Model: {MODEL_PATH}")
    print (f"🔗 Accessible on your network at: http://{local_ip}:{PORT}")
    print(f"🏠 Local testing URL: http://127.0.0.1:{PORT}")
    print("--------------------------------------------\n")
    print("🚀 Starting Uvicorn server...")

    # Run the FastAPI app using uvicorn
    uvicorn.run(app, host=HOST_IP, port=PORT) # reload=True removed for stability
