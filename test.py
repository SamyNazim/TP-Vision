import cv2
import numpy as np

print("=== Environment Verification ===")
print("OpenCV Version :", cv2.__version__)
print("NumPy Version  :", np.__version__)

# Simple test image popup
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), -1)
cv2.putText(img, "OpenCV Works!", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()