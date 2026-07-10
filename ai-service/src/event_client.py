import requests
from datetime import datetime, timezone

from config import DEVICE_ID, EVENT_API_URL, MODEL_VERSION

def send_prediction_event(gesture, confidence):
    payload = {
        "gesture": gesture,
        "confidence": confidence,
        "device_id": DEVICE_ID,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = requests.post(EVENT_API_URL, json=payload, timeout=3)
        response.raise_for_status()

        return True

    except requests.RequestException as error:
        print("Failed to send event:", error)
        return False
