
# pip install opencv-python
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

# Importa e converta para RGB
img = cv2.imread('./Satelite.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Filtro de ruído (bluring)
img_blur = cv2.blur(img, (5, 5))

# Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, 65, a, cv2.THRESH_BINARY_INV)

# preparando o "kernel"
kernel = np.ones((12, 12), np.uint8)


# operadores Morfologicos
img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Detecção borda com Canny (com open)
edges_open = cv2.Canny(image=img_open, threshold1=a/2, threshold2=a/2)


# contorno
contours, hierarchy = cv2.findContours(
    image=edges_open,
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                         color=(255, 0, 0), thickness=2)


# plot imagens
imagens = [img, img_blur, img_gray, thresh, img_open, final]
formatoX = math.ceil(len(imagens)**.5)
if (formatoX**2-len(imagens)) > formatoX:
    formatoY = formatoX-1
else:
    formatoY = formatoX
for i in range(len(imagens)):
    plt.subplot(formatoY, formatoX, i + 1)
    plt.imshow(imagens[i], 'gray')
    plt.xticks([]), plt.yticks([])
plt.show()
