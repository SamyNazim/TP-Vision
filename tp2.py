import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('green.jpg',cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("Image file not found. Please check the path and filename.")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# Créer bruit aléatoire
noise = np.random.normal(0,25,gray.shape)

# Appliquer bruit à l'image 
noisy = gray.astype(np.float32) + noise
noisy = np.clip(noisy,0, 255).astype(np.uint8)

# Afficher bruit et image
cv2.imshow("noise", noise)
cv2.imshow("noisy", noisy)

def blur_image(value):
    k = 2 * value + 1
    mean_filtered = cv2.blur(noisy,(k,k)) 
    cv2.imshow("Blur", mean_filtered)

cv2.namedWindow("Blur")
cv2.createTrackbar("Kernel size", "Blur",2,20,blur_image)
blur_image(2)

# mean (moyenne normale) gauss (moyenne pondérée)

def gauss_blur_image(value):
    k = 2 * value + 1
    gaussian = cv2.GaussianBlur(noisy, (k, k), 0)
    cv2.imshow("GaussianBlur", gaussian)


cv2.namedWindow("GaussianBlur")
cv2.createTrackbar("Kernel size", "GaussianBlur", 1, 20, gauss_blur_image)

gauss_blur_image(2)

# Filtre median : conserve mieux les contours

def median_blur_image(value):
    k = 2*value+1
    median = cv2.medianBlur(noisy, k)
    cv2.imshow("MedianBlur", median)

cv2.namedWindow("MedianBlur")
cv2.createTrackbar("Kernel size", "MedianBlur", 1, 20, median_blur_image)

median_blur_image(2)

# Histogramme (Fréquence des niveaux de gris)

hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure(figsize=[0,5])

plt.plot(hist)

plt.title("Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

# rendre image plus sombre: img_dark = img / 2

gray2 = (gray/2).astype(np.uint8)

hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

plt.figure(figsize=(5, 5))

plt.plot(hist2)

plt.title("Image Histogram (Dark)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

# rendre image plus claire: img_bright = img + 128

gray3 = (gray/2+127).astype(np.uint8)

hist3 = cv2.calcHist([gray3], [0], None, [256], [0, 256])

plt.figure(figsize=(5, 5))

plt.plot(hist3)

plt.title("Image Histogram (Bright)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

# equalize 
gray = (gray/2+50).astype(np.uint8)

hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure(figsize=[0,5])

plt.plot(hist)

plt.title("New Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

gray_equ = cv2.equalizeHist(gray)
hist2 = cv2.calcHist([gray_equ],[0],None,[256],[0,256])
plt.figure(figsize=[0,5])

plt.plot(hist)

plt.title("New Image Histogram (Equalized)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# new histogramme : probleme 0, 2, 4,, 6... peut etre corrigé avec la moyenne
# Filtre moyen (fonctions de base: boucles), filtre median, histogramme (AVEC FONCTIONS DE BASE)
