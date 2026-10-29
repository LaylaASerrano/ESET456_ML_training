from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(2)  # let auto-exposure settle
picam2.capture_file("test.jpg")
picam2.stop()
print("Saved test.jpg")

