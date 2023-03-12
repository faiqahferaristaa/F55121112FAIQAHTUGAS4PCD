import cv2

img = cv2.imread("foto.jpg", cv2.IMREAD_GRAYSCALE)

ksize = 3

# Median filter
median = cv2.medianBlur(img, ksize)

# Minimum filter
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
min_filter = cv2.erode(img, kernel)

# Maximum filter
max_filter = cv2.dilate(img, kernel)

# Show original and filtered images
cv2.imshow("Original Image", img)
cv2.imshow("Median Filter", median)
cv2.imshow("Minimum Filter", min_filter)
cv2.imshow("Maximum Filter", max_filter)

# Wait for key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()


 