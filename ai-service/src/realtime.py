import cv2
import numpy as np
import tensorflow as tf
import json

from config import IMAGE_WIDTH, IMAGE_HEIGHT, MODELS_DIR

MODEL_PATH = MODELS_DIR / "intivision_v1.keras"
LABELS_PATH = MODELS_DIR / "labels.json"

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera could not be opened.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Frame could not be read.")
            break

        frame_array = preprocess_frame(frame)

        predictions = model.predict(frame_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0])) 
        confidence = float(np.max(predictions[0]))

        predicted_label = labels[str(predicted_index)]


        text = f"{predicted_label} - {confidence * 100:.2f}%"

        cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        )

        cv2.imshow("IntiVision Realtime Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)    

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    frame_array = np.array(frame) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)

    return frame_array

if __name__ == "__main__":
    main()