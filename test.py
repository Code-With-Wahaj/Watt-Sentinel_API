import cv2
import numpy as np
import requests
import time

# Configuration
BACKEND_URL = "http://127.0.0.1:8000/detect"  # Change to your actual backend URL
SHOW_FPS = True
TARGET_FPS = 30  # Target frames per second


def send_frame_for_detection(frame):
    """Send frame to FastAPI backend for detection"""
    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)

    try:
        # Send to backend
        response = requests.post(
            BACKEND_URL,
            files={'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        )
        if response.status_code == 200:
            return response.json()['detections']
        else:
            print(f"Error: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame"""
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        confidence = detection['confidence']

        # Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Create label text
        label_text = f"{label}: {confidence:.2f}"

        # Calculate text position
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw background rectangle for text
        cv2.rectangle(frame,
                      (int(x1), int(y1) - text_height - 10),
                      (int(x1) + text_width, int(y1)),
                      (0, 255, 0), -1)

        # Put text
        cv2.putText(frame, label_text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables for FPS calculation
    prev_time = 0
    fps = 0

    print("Starting webcam detection. Press 'q' to quit.")

    while True:
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Send frame to backend for detection
        detections = send_frame_for_detection(frame)

        # Draw detections on frame
        draw_detections(frame, detections)

        # Display FPS if enabled
        if SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Real-Time Detection', frame)

        # Control frame rate
        key = cv2.waitKey(max(1, int(1000 / TARGET_FPS))) & 0xFF
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()