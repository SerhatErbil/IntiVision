import requests
from datetime import datetime, timezone


BACKEND_EVENT_URL = "http://localhost:8080/api/v1/events"
DEVICE_ID = "camera_01"
MODEL_VERSION = "v1"


def send_prediction_event(gesture, confidence):
    payload = {
        "gesture": gesture,
        "confidence": confidence,
        "device_id": DEVICE_ID,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = requests.post(BACKEND_EVENT_URL, json=payload, timeout=3)
        response.raise_for_status()

        return True

    except requests.RequestException as error:
        print("Failed to send event:", error)
        return False