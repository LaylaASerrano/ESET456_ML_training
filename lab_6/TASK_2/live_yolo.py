import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ----------------------------------------------------
# MODEL SETTINGS
# ----------------------------------------------------

MODEL_PATH = "yolo26n_320_ncnn_model"
IMAGE_SIZE = 320
CONFIDENCE = 0.25

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

model = YOLO(MODEL_PATH)

# ----------------------------------------------------
# CAMERA SETUP
# ----------------------------------------------------

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)

picam2.configure(config)
picam2.start()

print("YOLO26n CPU inference started.")
print("Press q to quit.")

# ----------------------------------------------------
# LIVE INFERENCE
# ----------------------------------------------------

while True:
    frame = picam2.capture_array()

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("YOLO26n - Raspberry Pi CPU", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------------------------------
# CLEANUP
# ----------------------------------------------------

picam2.stop()
cv2.destroyAllWindows()
