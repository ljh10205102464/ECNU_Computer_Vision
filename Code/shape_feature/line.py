import cv2
import numpy as np

# Read image 
img = cv2.imread('../../Figure/shape_feature/road.png', cv2.IMREAD_COLOR)
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 75, 200)
cv2.imwrite("edges_road.png", edges)
# Detect points that form a line
threshold = 100
lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold, minLineLength=1, maxLineGap=120)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imshow("Result Image", img)
cv2.imwrite("output_road.png", img)