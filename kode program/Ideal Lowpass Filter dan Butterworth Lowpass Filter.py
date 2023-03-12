import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('foto.jpg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(np.abs(fshift))

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
ideal_lp = np.zeros((rows, cols), np.uint8)
r = 60
ideal_lp[crow - r:crow + r, ccol - r:ccol + r] = 1

fshift_ideal_lp = fshift * ideal_lp

ideal_lp_img = np.fft.ifft2(np.fft.ifftshift(fshift_ideal_lp))
ideal_lp_img = np.abs(ideal_lp_img)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
d0 = 60
n = 2
butterworth_lp = np.zeros((rows, cols), np.uint8)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i - crow)**2 + (j - ccol)**2)
        butterworth_lp[i, j] = 1 / (1 + (d / d0)**(2*n))

fshift_butterworth_lp = fshift * butterworth_lp

butterworth_lp_img = np.fft.ifft2(np.fft.ifftshift(fshift_butterworth_lp))
butterworth_lp_img = np.abs(butterworth_lp_img)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Spektrum Amplitudo'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(ideal_lp_img, cmap='gray')
plt.title('Hasil Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(butterworth_lp_img, cmap='gray')
plt.title('Hasil Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])

cv2.imwrite('hasil_ideal_lp_filter.png', ideal_lp_img)
cv2.imwrite('hasil_butterworth_lp_filter.png', butterworth_lp_img)

plt.show()