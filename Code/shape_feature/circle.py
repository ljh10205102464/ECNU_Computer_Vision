import cv2
import numpy as np

# Read image as gray-scale
img = cv2.imread('../../Figure/shape_feature/eye.png', cv2.IMREAD_COLOR)
# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)

edges = cv2.Canny(img_blur, 1, 50)
cv2.imwrite("edges_eye.png", edges)

# Apply hough transform on the image
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=100, param2=15, minRadius=70, maxRadius=76)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imwrite("output_eye.png", img)