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
h, w = gray.shape
img_res_mean = np.zeros((h, w), dtype=np.uint8)

k = 3
p = k // 2

for i in range(p, h-p):
    for j in range(p, w-p):
        neighborhood = noisy[i-p:i+p+1, j-p:j+p+1]
        mean = np.mean(neighborhood, axis=(0, 1))
        img_res_mean[i, j] = mean

# Median Filter
img_res_median = np.zeros((h, w), dtype=np.uint8)

for i in range(p, h-p):
    for j in range(p, w-p):
        neighborhood = noisy[i-p:i+p+1, j-p:j+p+1]
        median = np.median(neighborhood, axis=(0, 1))
        img_res_median[i, j] = median

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

min = float(gray_dark.min())
max = float(gray_dark.max())

if max > min:
    gray_normalized = ((gray_dark.astype(np.float32) - min) * 255 / (max - min)).astype(np.uint8)
else:
    gray_normalized = gray_dark.copy()

hist_normalized = calculate_histogram(gray_normalized)

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
plt.imshow(img_res_mean, cmap='gray')
plt.title("Mean Filtered Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_res_median, cmap='gray')
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