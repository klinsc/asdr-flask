import cv2
import numpy as np

# Load image
image = cv2.imread("./images/mrb-Manorom2-sm-bh.jpg")

# Convert the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection
edges = cv2.Canny(gray, 50, 200, apertureSize=3)

# Perform a Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)

# Iterate over the output lines and draw them on the image
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Show result
cv2.imshow("Result Image", image)

# Save result
cv2.imwrite("./images/mrb-Manorom2-sm-bh-hough.jpg", image)

# Wait for key
cv2.waitKey(0)
