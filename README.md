# Watt Sentinel API – Security Anomaly Detection Backend

Backend inference service for the **Watt Sentinel Security Anomaly Detection System**.

This API powers the AI detection layer of the Watt Sentinel surveillance platform by receiving image frames from live CCTV streams through Flutter Web / Flutter Mobile applications, performing **YOLOv8-based anomaly and object detection**, and returning structured detections and alerts for persistent storage in Firebase.

---

## Overview

The Watt Sentinel API serves as the machine learning inference backend between the surveillance frontend applications and the trained YOLO anomaly detection model.

It enables real-time analysis of CCTV footage to identify suspicious or anomalous activities and generate actionable alerts.

---

## System Workflow

1. **Flutter Web / Flutter Mobile App** captures surveillance frames from live CCTV feeds
2. Frames are sent to the **FastAPI Detection Backend**
3. Backend performs **YOLOv8 Inference** using custom trained model
4. Detection results are returned as structured JSON
5. Flutter application stores processed results in **Firebase Database**, including:

   * Anomaly Detections
   * Detection Summaries
   * Security Scenarios / Events
   * Generated Alerts
   * Alert Resolution Metadata
   * User Who Handled Alert

---

## Features

* Real-time YOLOv8 anomaly/object detection
* FastAPI-powered REST backend
* Firebase-compatible structured JSON responses
* Returns:

  * Bounding Boxes
  * Labels
  * Confidence Scores
  * Detection Summary Counts
* Designed for Flutter Web & Mobile integration
* CORS Enabled for frontend communication
* LAN Accessible for local deployment/testing
* Includes standalone webcam/video testing client

---

## Tech Stack

* **Backend Framework:** FastAPI
* **ASGI Server:** Uvicorn
* **Computer Vision:** OpenCV
* **Deep Learning Model:** Ultralytics YOLOv8
* **Frontend Consumers:** Flutter Web / Flutter Mobile
* **Database:** Firebase Firestore / Realtime DB
* **Language:** Python

---

## System Architecture

```text id="arch1"
Flutter Web / Flutter App
        │
        ▼
 FastAPI Detection Backend
        │
        ▼
YOLOv8 Model Inference Engine
        │
        ▼
Structured Detection Response
        │
        ▼
 Firebase Database Storage
```

---

## API Response Format

```json id="resp1"
{
  "detections": [
    {
      "box": [120.5, 80.2, 300.4, 400.6],
      "label": "intruder",
      "confidence": 0.96
    }
  ],
  "detection_summary": {
    "intruder": 1
  }
}
```

---

## Project Structure

```bash id="struct1"
Watt-Sentinel_API/
│
├── yolo_server.py        # Main FastAPI inference backend
├── test.py               # Webcam/Video testing client for backend verification
├── best.pt               # Trained YOLOv8 model weights
├── requirements.txt      # Python dependencies
└── README.md
```

---

## File Descriptions

### `yolo_server.py`

Main production inference server responsible for:

* Loading trained YOLO model
* Exposing `/detect` API endpoint
* Running object/anomaly detection
* Returning prediction JSON responses

---

### `test.py`

Local testing utility used during development.

Capabilities:

* Opens webcam/video feed
* Captures frames continuously
* Sends frames to backend API
* Receives detection responses
* Draws bounding boxes and labels
* Displays live FPS counter

> **Note:** `test.py` is for development/testing only and is not part of the production Flutter pipeline.

---

## Installation

### Clone Repository

```bash id="clone1"
git clone https://github.com/Code-With-Wahaj/Watt-Sentinel_API.git
cd Watt-Sentinel_API
```

---

### Install Dependencies

```bash id="install1"
pip install -r requirements.txt
```

---

## Running the Backend Server

```bash id="run1"
python yolo_server.py
```

Server starts on:

```bash id="url1"
http://127.0.0.1:8000
```

---

## API Endpoint

### POST `/detect`

Accepts an image file and returns detection predictions.

### Request

**Form Data**

| Key  | Type       | Description      |
| ---- | ---------- | ---------------- |
| file | Image File | JPG / PNG / JPEG |

---

## Testing the Backend Locally

Run the webcam/video test client:

```bash id="test1"
python test.py
```

This will:

* Open webcam feed
* Send live frames to backend
* Render detections visually
* Show FPS counter

Press **Q** to quit.

---

## Flutter Frontend Integration

Your Flutter application should send surveillance frames to:

```dart id="flutter1"
http://YOUR_SERVER_IP:8000/detect
```

Returned predictions can then be stored in Firebase.

---

## Real-World Use Cases

This API powers the detection pipeline of the Watt Sentinel Security Platform for:

* Suspicious Activity Detection
* Intrusion Monitoring
* Real-Time Surveillance Analysis
* Security Event Logging
* Alert Generation & Management
* Historical Incident Analytics

---

## Future Enhancements

* WebSocket Streaming for Lower Latency
* Batch Frame Inference
* Multi-Camera Support
* Severity Classification System
* Push Notification Alerts
* Docker Deployment
* GPU Optimized Inference Pipeline
* Cloud Deployment Support

---

## Author

**Muhammad Wahaj Bin Aamir**
Flutter Developer • AI Enthusiast • Python Developer

* LinkedIn: [www.linkedin.com/in/muhammad-wahaj-bin-aamir](http://www.linkedin.com/in/muhammad-wahaj-bin-aamir)
* Email: [wahajaamir2@gmail.com](mailto:wahajaamir2@gmail.com)

---

## License

This project is licensed under the MIT License.
