import os
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('algiers.jpg')

if img is None:
    raise FileNotFoundError("Image file not found. Please check the path and filename.")

# Create noisy grayscale image

gray = (0.299 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 2]).astype(np.uint8)

noise = np.random.normal(0, 25, gray.shape)

noisy = gray.astype(np.float32) + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# Mean Filter

voisinage = 3

def filtreMoyenNVG(img):
    h, w = img.shape
    imgMoy = np.zeros(img.shape, np.uint8)

    for y in range(h):
        for x in range(w):
            if x < voisinage//2 or x >= w-voisinage//2 or y < voisinage//2 or y >= h-voisinage//2:
                imgMoy[y,x] = img[y,x]
            else:
                imgV = img[int(y-voisinage//2):int(y+voisinage//2)+1,
                           int(x-voisinage//2):int(x+voisinage//2)+1]

                moy = 0
                for yv in range(voisinage):
                    for xv in range(voisinage):
                        # int() avoids uint8 overflow when summing pixel values
                        moy += int(imgV[yv,xv])

                moy /= voisinage*voisinage

                imgMoy[y,x] = moy
                # imgMoy[y,x] = np.mean(imgV)

    return imgMoy

imgMoy = filtreMoyenNVG(noisy)

# Median Filter

def filtreMedianNVG(img):
    h, w = img.shape
    imgMed = np.zeros(img.shape, np.uint8)

    for y in range(h):
        for x in range(w):
            if x < voisinage//2 or x >= w-voisinage//2 or y < voisinage//2 or y >= h-voisinage//2:
                imgMed[y,x] = img[y,x]
            else:
                imgV = img[int(y-voisinage//2):int(y+voisinage//2)+1,
                           int(x-voisinage//2):int(x+voisinage//2)+1]

                t = np.zeros((voisinage*voisinage), np.uint8)

                for yv in range(voisinage):
                    for xv in range(voisinage):
                        t[yv*voisinage+xv] = imgV[yv,xv]

                t.sort()

                imgMed[y,x] = t[(voisinage*voisinage)//2]
                # imgMed[y,x] = np.median(imgV)

    return imgMed

imgMed = filtreMedianNVG(noisy)

# Histograms 

def calculate_histogram(image):
    hist = np.zeros(256, dtype=int)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            intensity = image[i, j]
            hist[intensity] += 1
    return hist

# Gray scale histogram

hist = calculate_histogram(gray)

# Darkened histogram

gray_dark = (gray / 2).astype(np.uint8)

hist_dark = calculate_histogram(gray_dark)

# Brightened histogram

gray_bright = np.clip(gray.astype(np.int16) + 128, 0, 255).astype(np.uint8)

hist_bright = calculate_histogram(gray_bright)

# Normalized histogram

imgNorm = np.zeros(gray_dark.shape, np.uint8)

h,w = gray_dark.shape

min = 255
max = 0

for y in range(h):
    for x in range(w):
        if gray_dark[y,x] > max:
            max = int(gray_dark[y,x])

        if gray_dark[y,x] < min:
            min = int(gray_dark[y,x])

if max > min:
    for y in range(h):
        for x in range(w):
            imgNorm[y,x] = (int(gray_dark[y,x]) - min) * 255 / (max - min)
else:
    imgNorm = gray_dark.copy()

hist_normalized = calculate_histogram(imgNorm)

# Smoothed histogram 

hist_smoothed = np.zeros(256, dtype=float)

for i in range(1, 255):
    hist_smoothed[i] = (hist_normalized[i-1] +
                        2 * hist_normalized[i] +
                        hist_normalized[i+1]) / 4

hist_smoothed[0] = (2 * hist_normalized[0] + hist_normalized[1]) / 3
hist_smoothed[255] = (hist_normalized[254] + 2 * hist_normalized[255]) / 3


# Display images

# Display original image, noise and noisy image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noise, cmap='gray')
plt.title("Random Noise")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy Image")
plt.axis("off")

plt.show()

# Display original, mean filtered and median filtered images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(imgMoy, cmap='gray')
plt.title("Mean Filtered Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(imgMed, cmap='gray')
plt.title("Median Filtered Image")
plt.axis("off")

plt.show()

# Display histograms

plt.figure(figsize=(12, 8))

# Grayscale histogram
plt.subplot(2, 2, 1)
plt.plot(hist)
plt.title("Grayscale Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

# Darkened histogram
plt.subplot(2, 2, 2)
plt.plot(hist_dark)
plt.title("Darkened Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

# Brightened histogram
plt.subplot(2, 2, 3)
plt.plot(hist_bright)
plt.title("Brightened Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

# Normalized histogram
plt.subplot(2, 2, 4)
plt.plot(hist_normalized)
plt.title("Dark Image Normalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

plt.tight_layout()
plt.show()

# Display normalized histogram and smoothed version

plt.figure(figsize=(12, 5))

# Normalized histogram 
plt.subplot(1, 2, 1)
plt.bar(range(256), hist_normalized, width=1)
plt.title("Normalized Dark Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

# Smoothed normalized histogram
plt.subplot(1, 2, 2)
plt.plot(hist_smoothed)
plt.title("Smoothed Normalized Dark Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()