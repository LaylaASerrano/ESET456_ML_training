import cv2
import os
from picamera2 import Picamera2
from ultralytics import YOLO

# ----------------------------------------------------
# MODEL SETTINGS
# ----------------------------------------------------

MODEL_PATH = os.path.expanduser("~/yolo26n_hailo_model")
IMAGE_SIZE = 320
CONFIDENCE = 0.25
TARGET_CLASS = "person"

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

model = YOLO(MODEL_PATH)

# ----------------------------------------------------
# OPTIONAL CLASS FILTERING
#
# Uncomment this section later in the lab to detect
# only the object selected by TARGET_CLASS.
# ----------------------------------------------------

# class_id = None
#
# for ID, name in model.names.items():
#     if name == TARGET_CLASS:
#         class_id = ID
#         break
#
# if class_id is None:
#     print("Class not found:", TARGET_CLASS)
#     exit()

# ----------------------------------------------------
# CAMERA SETUP
# ----------------------------------------------------

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)

picam2.configure(config)
picam2.start()

print("YOLO26n Hailo-8 inference started.")
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

        # Uncomment this line later in the lab
        # to enable object-class filtering.
        # classes=[class_id],

        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("YOLO26n - Hailo-8 AI HAT+", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------------------------------
# CLEANUP
# ----------------------------------------------------

picam2.stop()
cv2.destroyAllWindows()
