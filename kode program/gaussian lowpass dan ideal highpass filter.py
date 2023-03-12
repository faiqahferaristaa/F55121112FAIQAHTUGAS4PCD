import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('foto.jpg',0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Gaussian Lowpass Filter
rows, cols = img.shape
crow, ccol = rows//2 , cols//2
d = 60
sigma = 20
gaussian_mask = np.zeros((rows, cols), np.uint8)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i-crow)**2 + (j-ccol)**2)
        gaussian_mask[i,j] = np.exp(-((dist**2)/(2*(sigma**2))))
fshift_gaussian = fshift * gaussian_mask
f_gaussian = np.fft.ifftshift(fshift_gaussian)
img_gaussian = np.fft.ifft2(f_gaussian)
img_gaussian = np.abs(img_gaussian)

# Ideal Highpass Filter
r = 60
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (crow,ccol), r, 0, -1)
fshift_ideal = fshift * mask
f_ideal = np.fft.ifftshift(fshift_ideal)
img_ideal = np.fft.ifft2(f_ideal)
img_ideal = np.abs(img_ideal)

# Display images
plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img_gaussian, cmap = 'gray')
plt.title('Gaussian Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_ideal, cmap = 'gray')
plt.title('Ideal Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
