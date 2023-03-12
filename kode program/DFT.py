import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('foto.jpg', 0)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img, cmap='gray')
ax1.set_title('Gambar Asli')
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.set_title('Magnitudo Spektrum Frekuensi')
plt.show()

cv2.imwrite('hasil_dft.png', magnitude_spectrum)

cv2.waitKey(0)
cv2.destroyAllWindows()