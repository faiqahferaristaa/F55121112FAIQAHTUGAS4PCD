import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

img = cv2.imread('foto.jpg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(np.abs(fshift))

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
d0 = 30
n = 2
butterworth_hp = np.zeros((rows, cols), np.uint8)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i - crow)**2 + (j - ccol)**2)
        butterworth_hp[i, j] = 1 / (1 + (d0 / d)**(2*n))

fshift_butterworth_hp = fshift * butterworth_hp

butterworth_hp_img = np.fft.ifft2(np.fft.ifftshift(fshift_butterworth_hp))
butterworth_hp_img = np.abs(butterworth_hp_img)

ksize = 31
sigma = 5
gaussian_hp = 1 - gaussian_filter(np.ones_like(img), sigma=sigma, mode='constant')

fshift_gaussian_hp = fshift * gaussian_hp

gaussian_hp_img = np.fft.ifft2(np.fft.ifftshift(fshift_gaussian_hp))
gaussian_hp_img = np.abs(gaussian_hp_img)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Spektrum Amplitudo'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(butterworth_hp_img, cmap='gray')
plt.title('Hasil Butterworth Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(gaussian_hp_img, cmap='gray')
plt.title('Hasil Gaussian Highpass Filter'), plt.xticks([]), plt.yticks([])

cv2.imwrite('hasil_butterworth_hp_filter.png', butterworth_hp_img)
cv2.imwrite('hasil_gaussian_hp_filter.png', gaussian_hp_img)

plt.show()