import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Camera could not be opened")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    cv2.imshow("DeepGestic Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()