import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = './images/coin.jpg'
image = cv2.imread(image_path)

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯滤波
gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

print(gray_blurred.shape)
# 边缘检测
edges = cv2.Canny(gray_blurred, 30, 100)
cv2.imshow("", edges)
cv2.waitKey(0)

# 霍夫变换
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=0, maxRadius=0)


if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 轮廓
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)  # 圆心


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the result
plt.imshow(image_rgb)
plt.title('Detected Coin')
plt.axis('off')
plt.show()
